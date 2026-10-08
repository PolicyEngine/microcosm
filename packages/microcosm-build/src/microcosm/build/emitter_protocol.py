"""Transport contract shared by all local build publishing components."""

from typing import Final

ACTION_CLOSE: Final = "close"
ACTION_PING: Final = "ping"
LOCAL_ACKNOWLEDGEMENT_OK: Final = b"ok\n"
LOCAL_ACKNOWLEDGEMENT_ERROR: Final = b"error\n"
LOCAL_MESSAGE_DELIMITER: Final = b"\n"
LOCAL_PING_MESSAGE: Final = b'{"action":"ping"}\n'
MAX_LOCAL_MESSAGE_BYTES: Final = 1_048_576
LOCAL_SOCKET_READ_BYTES: Final = 65_536
LOCAL_ACKNOWLEDGEMENT_READ_BYTES: Final = 16
