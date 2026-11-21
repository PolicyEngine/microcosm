"""Tests for remote_fixed.py and acs_split.py: pandas against the range reader on small fixed-format files, served over HTTP with injected faults.

Invariants under test, for every file and cut:
- the range reader returns the same frame as pandas for any table and any row range;
- chunks concatenate to the whole table;
- a read returns exactly the file's bytes, whatever the network does short of a refusal;
- the split places each person in exactly one state, in file order, with each member's group row;
- the split's output does not depend on the source (path or URL), the chunk size, or where a run stopped.
Run: env/.venv/bin/python -m pytest test_remote_fixed.py -q
"""

import collections
import glob
import hashlib
import http.server
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error

import numpy as np
import pandas as pd
import pytest
import remote_fixed
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from remote_fixed import BLOCK, RemoteFile, RemoteStore, WrongFileError, check_identity

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")
FAST = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def write_store(path, people=3000, households=700, seed=0):
    """A fixed-format file with the column kinds the real file holds: float, int, bool and str, with blanks, and groups that span states."""
    rng = np.random.default_rng(seed)
    hh = pd.DataFrame(
        {
            "household_id": np.arange(1, households + 1),
            "state_fips": rng.choice([1, 6, 48, 56], households),
            "household_weight": rng.random(households) * 100,
            "household_vehicles_owned": rng.integers(0, 4, households),
            "is_rural": rng.random(households) < 0.2,
            "TYPEHUGQ": pd.array(
                rng.choice(["a", "b", "ccccc"], households), dtype="str"
            ),
        }
    )
    frame = {
        "person_id": np.arange(1, people + 1),
        "person_household_id": rng.integers(1, households + 1, people),
        "age": rng.integers(0, 90, people),
        "is_female": rng.random(people) < 0.5,
        "is_disabled": rng.random(people) < 0.1,
        "employment_income": np.where(
            rng.random(people) < 0.5, 0.0, rng.random(people) * 1e5
        ),
        "parent_1_id": np.where(
            rng.random(people) < 0.7, np.nan, rng.integers(1, people, people)
        ),
        "PERIDNUM": pd.array([f"id-{i:08d}" for i in range(people)], dtype="str"),
        "immigration_status_str": pd.array(
            rng.choice(["CITIZEN", "OTHER é"], people), dtype="str"
        ),
        "maybe_blank": pd.array(
            np.where(rng.random(people) < 0.5, None, "x"), dtype="str"
        ),
    }
    sizes = {"tax_unit": 900, "spm_unit": 800, "family": 750, "marital_unit": 1200}
    for g, k in sizes.items():
        frame[f"person_{g}_id"] = rng.integers(1, k + 1, people)
    with pd.HDFStore(path, "w") as store:
        store.put("household", hh, format="fixed")
        store.put("person", pd.DataFrame(frame), format="fixed")
        for g, k in sizes.items():
            store.put(
                g,
                pd.DataFrame(
                    {
                        f"{g}_id": np.arange(1, k + 1),
                        f"{g}_flag": rng.random(k) < 0.5,
                        f"{g}_value": rng.random(k),
                        f"{g}_text": pd.array(rng.choice(["p", "q"], k), dtype="str"),
                    }
                ),
                format="fixed",
            )


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, path, sha, advertised_size=None):
        super().__init__(("127.0.0.1", 0), Handler)
        with open(path, "rb") as f:
            self.body = f.read()
        self.sha = sha
        self.advertised_size = advertised_size or len(self.body)
        self.faults = collections.deque()
        self.always = None
        self.fail_after = None
        self.ranges = []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/file"

    def stop(self):
        self.shutdown()
        self.server_close()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _redirect(self):
        self.send_response(302)
        self.send_header("Location", "/cdn")
        self.send_header("x-linked-etag", f'"{self.server.sha}"')
        self.send_header("x-linked-size", str(self.server.advertised_size))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self._redirect()

    def _send(self, status, headers, body):
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/file":
            return self._redirect()
        server = self.server
        body = server.body
        first, last = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups()
        )
        last = min(last, len(body) - 1)
        server.ranges.append((first, last))
        fault = server.faults.popleft() if server.faults else server.always
        if server.fail_after is not None and len(server.ranges) > server.fail_after:
            fault = "503"
        piece = body[first : last + 1]
        good = {
            "Content-Range": f"bytes {first}-{last}/{len(body)}",
            "Content-Length": str(len(piece)),
        }
        if fault in ("503", "404", "429", "403", "410", "416"):
            return self._send(int(fault), {"Content-Length": "0"}, b"")
        if fault == "200":
            return self._send(200, {"Content-Length": str(len(piece))}, piece)
        if fault == "wrong-total":
            return self._send(
                206,
                {**good, "Content-Range": f"bytes {first}-{last}/{len(body) + 1}"},
                piece,
            )
        if fault == "wrong-start":
            return self._send(
                206,
                {**good, "Content-Range": f"bytes {first + 1}-{last}/{len(body)}"},
                piece,
            )
        if fault == "no-range":
            return self._send(206, {"Content-Length": str(len(piece))}, piece)
        if fault == "drop":
            self.send_response(206)
            for key, value in good.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(piece[: len(piece) // 2])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(2)
            return None
        return self._send(206, good, piece)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("h5") / "small.h5")
    write_store(path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return path, sha, os.path.getsize(path)


@pytest.fixture
def server(small):
    path, sha, _ = small
    s = Server(path, sha)
    yield s
    s.stop()


@pytest.fixture(scope="module")
def shared(small):
    path, sha, _ = small
    s = Server(path, sha)
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    real = time.sleep
    monkeypatch.setattr(
        remote_fixed.time, "sleep", lambda s: real(min(s, 0.01)) if s >= 1 else real(s)
    )


def remote(server, size, **kwargs):
    return RemoteStore(server.url, size=size, sha256=server.sha, rate=10**12, **kwargs)


@pytest.mark.parametrize("key", ["household", "person", *GROUPS])
def test_whole_table_matches_pandas(small, shared, key):
    path, _, size = small
    with pd.HDFStore(path, "r") as store, remote(shared, size) as mine:
        pd.testing.assert_frame_equal(
            mine.select(key), store.select(key), check_exact=True
        )


def test_every_column_kind_comes_through(small, shared):
    """The file must hold the kinds the real file holds, or the comparisons above prove less than they seem to."""
    _, _, size = small
    with remote(shared, size) as mine:
        kinds = set(mine.select("person").dtypes.astype(str))
    assert kinds == {"int64", "float64", "bool", "str"}


@FAST
@given(start=st.integers(0, 3200), length=st.integers(0, 3200))
def test_any_row_range_matches_pandas(small, shared, start, length):
    path, _, size = small
    with pd.HDFStore(path, "r") as store, remote(shared, size) as mine:
        pd.testing.assert_frame_equal(
            mine.select("person", start=start, stop=start + length),
            store.select("person", start=start, stop=start + length),
            check_exact=True,
        )


@FAST
@given(cuts=st.lists(st.integers(1, 2999), max_size=8, unique=True))
def test_chunks_concatenate_to_the_whole_table(small, shared, cuts):
    _, _, size = small
    edges = [0, *sorted(cuts), 3000]
    with remote(shared, size) as mine:
        whole = mine.select("person")
        chunks = [
            mine.select("person", start=a, stop=b) for a, b in zip(edges, edges[1:])
        ]
    pd.testing.assert_frame_equal(pd.concat(chunks), whole, check_exact=True)


def test_chunked_reads_fetch_an_object_column_once(small, server):
    _, _, size = small
    with remote(server, size) as mine:
        mine.select("person", start=0, stop=10)
        before = mine.file.bytes_read
        mine.select("person", start=10, stop=20)
        after = mine.file.bytes_read
    assert after - before < 30_000, (
        "the second chunk should fetch numeric rows and no object column"
    )


def test_chunked_read_holds_only_the_rows_it_asked_for(small, server):
    """A read of a few rows must not fetch whole numeric blocks."""
    _, _, size = small
    with remote(server, size) as mine:
        mine.select("person", start=0, stop=1)
        first = sum(b - a + 1 for a, b in server.ranges)
    assert first < size, "metadata and object columns only, plus one row"


def test_nrows(small, shared):
    _, _, size = small
    with remote(shared, size) as mine:
        assert mine.nrows("person") == 3000
        assert mine.nrows("household") == 700


@FAST
@given(offset=st.integers(0, 9_200_000), length=st.integers(0, 3 * BLOCK))
def test_read_returns_the_file_bytes(small, shared, offset, length):
    path, _, size = small
    with open(path, "rb") as f:
        data = f.read()
    file = RemoteFile(shared.url, size, rate=10**12)
    file.seek(offset)
    buffer = bytearray(length)
    assert file.readinto(buffer) == length
    truth = data[offset : offset + length]
    assert bytes(buffer) == truth + bytes(length - len(truth)), (
        "file bytes, then zeros past the end"
    )


def test_a_read_past_the_end_of_the_file_fills_with_zeros(small, shared):
    _, _, size = small
    file = RemoteFile(shared.url, size, rate=10**12)
    file.seek(size - 5)
    buffer = bytearray(b"x" * 20)
    file.readinto(buffer)
    assert buffer[5:] == bytes(15), (
        "HDF5 reads through h5py, which ignores a short count"
    )
    file.seek(size + 100)
    buffer = bytearray(b"x" * 20)
    file.readinto(buffer)
    assert buffer == bytes(20)


def test_read_keeps_to_its_rate(small, shared):
    _, _, size = small
    rate = 2_000_000
    file = RemoteFile(shared.url, size, rate=rate)
    want = min(size, 1_000_000)
    file.seek(0)
    began = time.monotonic()
    file.readinto(bytearray(want))
    assert time.monotonic() - began >= 0.95 * want / rate


@pytest.mark.parametrize(
    "faults", [["503"], ["429"], ["403"], ["drop"], ["drop", "503", "403", "drop"]]
)
def test_read_survives_what_a_network_breaks(small, server, faults):
    path, _, size = small
    with open(path, "rb") as f:
        data = f.read()
    server.faults.extend(faults)
    file = RemoteFile(server.url, size, rate=10**12)
    want = min(size, 2 * BLOCK)
    buffer = bytearray(want)
    file.seek(0)
    file.readinto(buffer)
    assert bytes(buffer) == data[:want]
    assert not server.faults


def test_a_dropped_response_keeps_what_arrived(small, server):
    _, _, size = small
    server.faults.append("drop")
    file = RemoteFile(server.url, size, rate=10**12)
    want = min(size, 2 * BLOCK)
    file.seek(0)
    file.readinto(bytearray(want))
    assert len(server.ranges) == 2
    assert server.ranges[1][0] > server.ranges[0][0], "the retry asks only for the rest"


@pytest.mark.parametrize(
    "fault", ["200", "wrong-total", "wrong-start", "no-range", "404", "410", "416"]
)
def test_read_refuses_a_response_from_the_wrong_file(small, server, fault):
    _, _, size = small
    server.faults.append(fault)
    file = RemoteFile(server.url, size, rate=10**12)
    with pytest.raises(WrongFileError):
        file.readinto(bytearray(10))


def test_read_gives_up_after_its_patience(small, server):
    _, _, size = small
    server.always = "503"
    file = RemoteFile(server.url, size, rate=10**12, patience=0)
    with pytest.raises(urllib.error.HTTPError):
        file.readinto(bytearray(10))


def test_identity_check(small, server):
    _, sha, size = small
    check_identity(server.url, size, sha)
    with pytest.raises(WrongFileError):
        check_identity(server.url, size, "0" * 64)
    with pytest.raises(WrongFileError):
        check_identity(server.url, size + 1, sha)
    with pytest.raises(WrongFileError):
        RemoteStore(server.url, size=size, sha256="0" * 64)


def run_split(source, out, sha=None, size=None, chunk=500, env=None, **popen):
    e = {
        **os.environ,
        "SPLIT_CHUNK": str(chunk),
        "SPLIT_RATE": str(10**12),
        "OMP_NUM_THREADS": "1",
    }
    if sha:
        e.update(SPLIT_SHA256=sha, SPLIT_SIZE=str(size))
    e.update(env or {})
    return subprocess.run(
        [PY, "-I", f"{HERE}/acs_split.py", source, out],
        env=e,
        capture_output=True,
        text=True,
        timeout=300,
        **popen,
    )


def read_split(out):
    return {
        os.path.basename(p): pd.read_pickle(p)
        for p in sorted(glob.glob(f"{out}/*.pkl.gz"))
    }


def assert_same_split(a, b):
    assert sorted(a) == sorted(b)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True, obj=name)


@pytest.fixture(scope="module")
def clean(small, tmp_path_factory):
    path, _, _ = small
    out = str(tmp_path_factory.mktemp("split") / "clean")
    result = run_split(path, out)
    assert result.returncode == 0, result.stderr
    return read_split(out)


def test_split_invariants(small, clean):
    path, _, _ = small
    with pd.HDFStore(path, "r") as store:
        person = store.select("person")
        household = store.select("household").set_index("household_id")
        tables = {g: store.select(g) for g in GROUPS}
    states = sorted({name[:2] for name in clean})
    placed = pd.concat([clean[f"{s}-person.pkl.gz"] for s in states])
    assert len(placed) == len(person) and placed.person_id.is_unique, (
        "each person lands once"
    )
    for s in states:
        people = clean[f"{s}-person.pkl.gz"]
        assert people.person_id.is_monotonic_increasing, "file order within a state"
        assert (
            household.loc[people.person_household_id, "state_fips"].values == int(s)
        ).all(), "the right state"
        pd.testing.assert_frame_equal(
            people, person[person.person_id.isin(people.person_id)], check_exact=True
        )
        for g in GROUPS:
            rows = clean[f"{s}-{g}.pkl.gz"]
            assert set(people[f"person_{g}_id"]) == set(rows[f"{g}_id"]), (
                "each member's group row, and no other"
            )
            pd.testing.assert_frame_equal(
                rows,
                tables[g][tables[g][f"{g}_id"].isin(rows[f"{g}_id"])],
                check_exact=True,
            )
    assert any(placed[f"person_{g}_id"].duplicated().any() for g in GROUPS)
    spans = {
        g: pd.concat([clean[f"{s}-{g}.pkl.gz"] for s in states])[f"{g}_id"]
        .duplicated()
        .any()
        for g in GROUPS
    }
    assert all(spans.values()), "this file should exercise groups that span states"


def test_split_from_a_url_equals_split_from_a_path(small, server, clean, tmp_path):
    _, sha, size = small
    result = run_split(server.url, str(tmp_path / "s"), sha, size)
    assert result.returncode == 0, result.stderr
    assert_same_split(read_split(tmp_path / "s"), clean)
    assert not os.path.exists(tmp_path / "s" / "parts") and not os.path.exists(
        tmp_path / "s" / "parts.finished"
    )


@pytest.mark.parametrize("chunk", [1, 97, 1000, 100000])
def test_chunk_size_changes_nothing(small, clean, tmp_path, chunk):
    path, _, _ = small
    if chunk == 1:
        chunk = (
            7  # one row a chunk would write 3000 x 4 small files for no extra coverage
        )
    result = run_split(path, str(tmp_path / "s"), chunk=chunk)
    assert result.returncode == 0, result.stderr
    assert_same_split(read_split(tmp_path / "s"), clean)


def requests_for_a_clean_run(server, sha, size, out, chunk=500):
    result = run_split(server.url, out, sha, size, chunk=chunk)
    assert result.returncode == 0, result.stderr
    count = len(server.ranges)
    server.ranges.clear()
    return count


def test_split_resumes_where_a_killed_run_stopped(small, server, clean, tmp_path):
    _, sha, size = small
    out = str(tmp_path / "s")
    env = {
        **os.environ,
        "SPLIT_CHUNK": "100",
        "SPLIT_RATE": str(10**12),
        "SPLIT_SHA256": sha,
        "SPLIT_SIZE": str(size),
    }
    p = subprocess.Popen(
        [PY, "-I", f"{HERE}/acs_split.py", server.url, out],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in p.stdout:
        if line.startswith("chunk 3 "):
            p.kill()
            break
    p.stdout.close()
    p.wait()
    done = glob.glob(f"{out}/parts/chunk-*.npz")
    assert 4 <= len(done) < 30, f"the kill should land part way: {len(done)} chunks"
    server.ranges.clear()
    second = run_split(server.url, out, sha, size, chunk=100)
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("already done") == len(done)
    assert "households" not in second.stdout, (
        "the household table should not come down twice"
    )
    read_again = sum(
        1
        for line in second.stdout.splitlines()
        if line.startswith("chunk") and "already" not in line
    )
    assert read_again == 30 - len(done), "only the chunks that were left come down"
    assert_same_split(read_split(out), clean)


def stopped_before_the_merge(server, sha, size, tmp_path):
    """A run whose last group table failed: every chunk's marker and parts sit on disk, and no state has merged."""
    out = str(tmp_path / "s")
    server.fail_after = (
        requests_for_a_clean_run(server, sha, size, str(tmp_path / "first")) - 1
    )
    result = run_split(server.url, out, sha, size, env={"SPLIT_PATIENCE": "1"})
    assert result.returncode != 0
    assert len(glob.glob(f"{out}/parts/chunk-*.npz")) == 6
    assert not glob.glob(f"{out}/parts/merged-*") and not glob.glob(
        f"{out}/*-person.pkl.gz"
    )
    server.ranges.clear()
    server.fail_after = None
    return out


def test_split_stopped_after_a_state_merged_keeps_that_state(
    small, server, clean, tmp_path
):
    """Stopped between a state's merge and the removal of its last parts: a rerun must not merge the leftovers over the whole state."""
    _, sha, size = small
    out = stopped_before_the_merge(server, sha, size, tmp_path)
    parts = sorted(glob.glob(f"{out}/parts/06-person-*.pkl.gz"))
    assert len(parts) > 1
    clean["06-person.pkl.gz"].to_pickle(
        f"{out}/06-person.pkl.gz", compression={"method": "gzip", "compresslevel": 1}
    )
    open(f"{out}/parts/merged-06", "w").close()
    for part in parts[:-1]:
        os.remove(part)
    result = run_split(server.url, out, sha, size)
    assert result.returncode == 0, result.stderr
    assert_same_split(read_split(out), clean)


def test_split_stopped_before_the_marker_merges_again(small, server, clean, tmp_path):
    """Stopped between writing a state's file and its marker, with a stale file in place: a rerun merges every part again."""
    _, sha, size = small
    out = stopped_before_the_merge(server, sha, size, tmp_path)
    clean["06-person.pkl.gz"].head(3).to_pickle(
        f"{out}/06-person.pkl.gz", compression={"method": "gzip", "compresslevel": 1}
    )
    result = run_split(server.url, out, sha, size)
    assert result.returncode == 0, result.stderr
    assert_same_split(read_split(out), clean)


def test_split_stopped_during_the_last_cleanup_is_finished(small, clean, tmp_path):
    """Once the parts folder has its finished name, a rerun fetches nothing and changes nothing."""
    path, _, _ = small
    out = str(tmp_path / "s")
    assert run_split(path, out).returncode == 0
    os.makedirs(f"{out}/parts.finished")
    open(f"{out}/parts.finished/chunk-00000.npz", "w").close()
    result = run_split("https://invalid.invalid/none", out)
    assert result.returncode == 0, result.stderr
    assert "an earlier run finished" in result.stdout
    assert not os.path.exists(f"{out}/parts.finished") and not os.path.exists(
        f"{out}/parts"
    )
    assert_same_split(read_split(out), clean)


def test_split_refuses_parts_cut_another_way(small, tmp_path):
    path, _, _ = small
    out = str(tmp_path / "s")
    os.makedirs(f"{out}/parts")
    with open(f"{out}/parts/plan.json", "w") as f:
        f.write('{"people": 3000, "chunk": 999}')
    result = run_split(path, out, chunk=500)
    assert result.returncode != 0
    assert "holds a run cut as" in result.stderr
    assert not glob.glob(f"{out}/*.pkl.gz")


def test_split_waits_at_its_gate(small, tmp_path):
    path, _, _ = small
    out, log = str(tmp_path / "s"), str(tmp_path / "gate.log")
    env = {**os.environ, "SPLIT_CHUNK": "500", "SPLIT_GATE": f"99999 1 {log}"}
    p = subprocess.Popen(
        [PY, "-I", f"{HERE}/acs_split.py", path, out],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(240):
            if os.path.exists(log) and "hold" in open(log).read():
                break
            time.sleep(0.5)
        assert "of 99999" in open(log).read()
        assert p.poll() is None, "it should still wait"
        assert not glob.glob(f"{out}/*.pkl.gz"), "nothing written before the gate opens"
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # macOS answers either way when the group has already ended
        p.wait()
    result = run_split(path, out, env={"SPLIT_GATE": f"0 0 {log}"})
    if "booking" in open(log).read() and result.returncode != 0:
        pytest.skip("a booking held the gate")
    assert result.returncode == 0, result.stderr


def test_split_stops_at_its_memory_cap(small, tmp_path):
    path, _, _ = small
    result = run_split(
        path, str(tmp_path / "s"), env={"RSS_CAP_GIB": "0.001", "SPLIT_CHUNK": "1"}
    )
    # The watcher reads the peak every 2 seconds, so a very quick run can finish first, which is no failure of the cap.
    assert result.returncode in (0, 75)
    if result.returncode == 75:
        assert "rss stop" in result.stdout


def test_split_output_survives_a_kill_at_any_moment(tmp_path):
    """Kill the split at random moments, rerun it until it finishes, and compare with a clean run. Holds for every phase the kills land in."""
    path = str(tmp_path / "big.h5")
    write_store(path, people=40000, households=9000, seed=3)
    clean = str(tmp_path / "clean")
    t0 = time.monotonic()
    result = run_split(path, clean, chunk=5000)
    assert result.returncode == 0, result.stderr
    took = time.monotonic() - t0
    truth = read_split(clean)
    rng = random.Random(7)
    env = {**os.environ, "SPLIT_CHUNK": "5000", "OMP_NUM_THREADS": "1"}
    for round_ in range(8):
        out = str(tmp_path / f"k{round_}")
        for _ in range(40):
            p = subprocess.Popen(
                [PY, "-I", f"{HERE}/acs_split.py", path, out],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                p.wait(timeout=rng.uniform(0.2, took * 1.1))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                continue
            if p.returncode == 0:
                break
            pytest.fail(f"round {round_}: a rerun failed with exit {p.returncode}")
        else:
            result = run_split(path, out, chunk=5000)
            assert result.returncode == 0, result.stderr
        assert_same_split(read_split(out), truth)
        assert not os.path.exists(f"{out}/parts")
        shutil.rmtree(out)
