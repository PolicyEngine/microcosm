"""Read pandas fixed-format tables from a remote H5 file through HTTP range requests, with no local copy.

RemoteFile gives h5py a seekable file that fetches only the bytes asked of it, at no more than `rate` bytes a second. It checks the file's identity
once when it opens, and every response's range and total size, and it retries what a network can break. RemoteStore reads a table through pandas' own
fixed-format reader and replaces only the step that takes an array from PyTables, which cannot open a remote file.

    with RemoteStore(url, size=..., sha256=...) as store:
        n = store.nrows("person")
        people = store.select("person", start=0, stop=1000)
"""

import collections
import http.client
import io
import pickle
import time
import urllib.error
import urllib.request

import h5py
import numpy as np
import pandas as pd
from pandas.io.pytables import FrameFixed

BLOCK = 2**18  # small reads share cached blocks of this size
CACHE_BLOCKS = 256  # 64 MiB of them
STEP = 2**26  # one request asks for at most this much
PIECE = 2**16  # the pace check runs after each piece of a response
RATE = 3_000_000  # bytes a second
PATIENCE = 7200  # seconds a read may fail without any progress before it gives up
TIMEOUT = 120  # seconds one socket operation may stall
AGENT = "policyengine-state-child-spm"


class WrongFileError(RuntimeError):
    """The server answered, and the answer says this is not the expected file. A retry cannot help."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def check_identity(url, size, sha256, patience=PATIENCE):
    """Read the file's size and content hash from the pinned URL's own response, before it redirects to storage."""
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": AGENT})
    began = time.monotonic()
    while True:
        try:
            try:
                headers = opener.open(request, timeout=TIMEOUT).headers
            except urllib.error.HTTPError as error:
                if error.code >= 500:
                    raise
                headers = error.headers
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            if time.monotonic() - began > patience:
                raise
            print("identity check retry:", repr(error)[:200], flush=True)
            time.sleep(5)
    etag = (headers.get("x-linked-etag") or "").strip('"')
    linked = headers.get("x-linked-size")
    if etag != sha256 or linked != str(size):
        raise WrongFileError(
            f"expected sha256 {sha256} and size {size}, the URL reports {etag!r} and {linked!r}"
        )


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size, rate=RATE, patience=PATIENCE):
        self.url, self.size, self.rate, self.patience = url, size, rate, patience
        self.pos = self.bytes_read = self.requests = 0
        self.blocks = collections.OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {
            io.SEEK_SET: offset,
            io.SEEK_CUR: self.pos + offset,
            io.SEEK_END: self.size + offset,
        }[whence]
        return self.pos

    def readinto(self, buffer):
        """h5py ignores a short count, so this fills the whole buffer: file bytes, then zeros past the end, as a local file read gives HDF5."""
        view = memoryview(buffer).cast("B")
        want = max(0, min(len(view), self.size - self.pos))
        if want >= BLOCK:
            self._fetch(self.pos, view[:want])
        else:
            done = 0
            while done < want:
                index, offset = divmod(self.pos + done, BLOCK)
                block = self._block(index)
                n = min(want - done, len(block) - offset)
                view[done : done + n] = block[offset : offset + n]
                done += n
        view[want:] = bytes(len(view) - want)
        self.pos += want
        return len(view)

    def _block(self, index):
        if index in self.blocks:
            self.blocks.move_to_end(index)
            return self.blocks[index]
        start = index * BLOCK
        block = bytearray(min(BLOCK, self.size - start))
        self._fetch(start, memoryview(block))
        self.blocks[index] = block
        while len(self.blocks) > CACHE_BLOCKS:
            self.blocks.popitem(last=False)
        return block

    def _fetch(self, start, view):
        """Fill view with the file's bytes from start. Keeps what arrived when a request fails, and asks again for the rest."""
        done, failing_since, delay = 0, None, 1.0
        while done < len(view):
            stop = min(done + STEP, len(view))
            before = done
            try:
                done = self._request(start, view, done, stop)
                failing_since, delay = None, 1.0
            except WrongFileError:
                raise
            except urllib.error.HTTPError as error:
                if error.code in (404, 410, 416):
                    raise WrongFileError(
                        f"HTTP {error.code} on the pinned URL"
                    ) from error
                done, failing_since, delay = self._wait(
                    error, view, start, before, failing_since, delay
                )
            except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
                done, failing_since, delay = self._wait(
                    error, view, start, before, failing_since, delay
                )

    def _wait(self, error, view, start, before, failing_since, delay):
        done = getattr(error, "done", before)
        now = time.monotonic()
        if done > before:
            failing_since, delay = None, 1.0
        failing_since = failing_since or now
        if now - failing_since > self.patience:
            raise error
        print(
            f"read retry at {start + done} after {delay:.0f}s:",
            repr(error)[:200],
            flush=True,
        )
        time.sleep(delay)
        return done, failing_since, min(delay * 2, 300.0)

    def _request(self, start, view, done, stop):
        first, last = start + done, start + stop - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={first}-{last}", "User-Agent": AGENT}
        )
        self.requests += 1
        got, began = 0, time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.status != 206:
                    raise WrongFileError(
                        f"expected a 206 for bytes {first}-{last}, got {response.status}"
                    )
                reported = response.headers.get("Content-Range")
                if reported != f"bytes {first}-{last}/{self.size}":
                    raise WrongFileError(
                        f"asked for bytes {first}-{last} of {self.size}, got {reported!r}"
                    )
                while done < stop:
                    n = response.readinto(view[done : min(done + PIECE, stop)])
                    if not n:
                        raise ConnectionError(
                            f"response ended early at byte {start + done}"
                        )
                    done += n
                    got += n
                    self.bytes_read += n
                    pause = got / self.rate - (time.monotonic() - began)
                    if pause > 0:
                        time.sleep(pause)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            error.done = done
            raise
        return done


class _Attrs:
    """A dataset's attributes as PyTables gives them to pandas: text as str, and the pickled None that PyTables writes as None."""

    def __init__(self, attrs):
        self._attrs = attrs

    def __contains__(self, name):
        return name in self._attrs

    def __getattr__(self, name):
        if name not in self._attrs:
            raise AttributeError(name)
        value = self._attrs[name]
        if isinstance(value, np.bytes_):
            return None if value == b"N." else value.decode()
        return value


class _Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self._v_attrs = _Attrs(dataset.attrs)
        self._v_name = dataset.name.rsplit("/", 1)[-1]

    def __getitem__(self, key):
        return self.dataset[key]


class _Group:
    def __init__(self, group):
        self._group = group
        self._v_attrs = _Attrs(group.attrs)

    def __getattr__(self, name):
        if name not in self._group:
            raise AttributeError(name)
        return _Node(self._group[name])


class _RemoteFrame(FrameFixed):
    def __init__(self, store, group):
        self.parent, self.group = store, group
        self.encoding, self.errors = "UTF-8", "strict"

    def read_array(self, key, start=None, stop=None):
        """pandas' GenericFixed.read_array for the arrays these files hold, read through h5py. Refuses the kinds it does not mirror."""
        node = getattr(self.group, key)
        attrs, dataset = node._v_attrs, node.dataset
        value_type = getattr(attrs, "value_type", None)
        kind = dataset.id.get_type().get_class()
        if kind == h5py.h5t.VLEN:
            ret = self.parent.objects(
                dataset, keep=start is not None or stop is not None
            )[start:stop]
            if value_type is not None:
                ret = pd.array(ret, dtype=value_type)
        elif value_type is not None or "shape" in attrs:
            raise NotImplementedError(
                f"{dataset.name}: value_type {value_type}, empty or date arrays read only through pandas"
            )
        else:
            ret = dataset[start:stop]
            if kind == h5py.h5t.BITFIELD:
                ret = ret.astype(bool)
        return ret.T if getattr(attrs, "transposed", False) else ret


class RemoteStore:
    """The parts of pandas.HDFStore that a chunked read of fixed-format tables needs."""

    def __init__(self, source, size=None, sha256=None, rate=RATE, patience=PATIENCE):
        if isinstance(source, str):
            check_identity(source, size, sha256, patience)
            source = RemoteFile(source, size, rate, patience)
        self.file = source
        self.h5 = h5py.File(self.file, "r")
        self._objects = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.h5.close()

    def nrows(self, key):
        return self.h5[key]["axis1"].shape[0]

    def select(self, key, start=None, stop=None):
        frame = _RemoteFrame(self, _Group(self.h5[key]))
        if not frame.infer_axes():
            raise KeyError(key)
        return frame.read(start=start, stop=stop)

    def objects(self, dataset, keep):
        """A whole object column, as PyTables reads it. A chunked read slices it many times, so it stays in memory then."""
        if dataset.name in self._objects:
            return self._objects[dataset.name]
        value = pickle.loads(dataset[0].tobytes())
        if keep:
            self._objects[dataset.name] = value
        return value
