from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import socketserver
import struct
import threading
import time
from typing import Dict, Optional, Tuple


MAGIC = b"FDSP"
PROTOCOL_VERSION = 1
PREFIX = struct.Struct("!4sBIQ")
DEFAULT_MAX_MESSAGE_BYTES = 128 * 1024 * 1024


class RpcProtocolError(RuntimeError):
    """Raised when an RPC frame violates the wire protocol."""


@dataclass(frozen=True)
class RpcFrame:
    message_type: str
    metadata: Dict[str, object]
    payload: bytes
    wire_bytes: int


@dataclass(frozen=True)
class RpcResponse:
    message_type: str
    metadata: Dict[str, object]
    payload: bytes = b""
    delay_s: float = 0.0
    packet_id: str = ""


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly size bytes or fail on a closed connection."""

    if int(size) < 0:
        raise RpcProtocolError("Cannot receive a negative byte count.")
    chunks = bytearray()
    while len(chunks) < int(size):
        block = sock.recv(int(size) - len(chunks))
        if not block:
            raise RpcProtocolError(
                "Connection closed before the complete frame arrived."
            )
        chunks.extend(block)
    return bytes(chunks)


def encode_frame(
    message_type: str,
    metadata: Dict[str, object],
    payload: bytes = b"",
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> bytes:
    header = json.dumps(
        {
            "message_type": str(message_type),
            "metadata": dict(metadata),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload = bytes(payload)
    total = len(header) + len(payload)
    if total > int(max_message_bytes):
        raise RpcProtocolError(
            f"RPC message exceeds max_message_bytes={max_message_bytes}."
        )
    prefix = PREFIX.pack(
        MAGIC,
        PROTOCOL_VERSION,
        len(header),
        len(payload),
    )
    return prefix + header + payload


def send_frame(
    sock: socket.socket,
    message_type: str,
    metadata: Dict[str, object],
    payload: bytes = b"",
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> int:
    frame = encode_frame(
        message_type,
        metadata,
        payload,
        max_message_bytes=max_message_bytes,
    )
    sock.sendall(frame)
    return len(frame)


def receive_frame(
    sock: socket.socket,
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> RpcFrame:
    prefix = recv_exact(sock, PREFIX.size)
    magic, version, header_length, payload_length = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise RpcProtocolError("Invalid RPC magic.")
    if int(version) != PROTOCOL_VERSION:
        raise RpcProtocolError(
            f"Unsupported RPC protocol version={version}."
        )
    total = int(header_length) + int(payload_length)
    if int(header_length) <= 0:
        raise RpcProtocolError("RPC header cannot be empty.")
    if total > int(max_message_bytes):
        raise RpcProtocolError(
            f"RPC message exceeds max_message_bytes={max_message_bytes}."
        )
    header_bytes = recv_exact(sock, int(header_length))
    payload = recv_exact(sock, int(payload_length))
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RpcProtocolError("RPC header is not valid JSON.") from exc
    if not isinstance(header, dict):
        raise RpcProtocolError("RPC header must be a JSON object.")
    message_type = header.get("message_type")
    metadata = header.get("metadata")
    if not isinstance(message_type, str) or not message_type:
        raise RpcProtocolError("RPC message_type is missing.")
    if not isinstance(metadata, dict):
        raise RpcProtocolError("RPC metadata must be a JSON object.")
    return RpcFrame(
        message_type=message_type,
        metadata=metadata,
        payload=payload,
        wire_bytes=PREFIX.size + int(header_length) + int(payload_length),
    )


def rpc_call(
    host: str,
    port: int,
    *,
    message_type: str,
    metadata: Dict[str, object],
    payload: bytes = b"",
    timeout_s: float,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> Tuple[RpcFrame, int, float]:
    """Perform one request/response TCP RPC using a fresh connection."""

    started = time.monotonic()
    with socket.create_connection(
        (str(host), int(port)),
        timeout=float(timeout_s),
    ) as sock:
        sock.settimeout(float(timeout_s))
        request_wire_bytes = send_frame(
            sock,
            message_type,
            metadata,
            payload,
            max_message_bytes=max_message_bytes,
        )
        response = receive_frame(
            sock,
            max_message_bytes=max_message_bytes,
        )
    elapsed = time.monotonic() - started
    if response.message_type == "ERROR":
        raise RpcProtocolError(
            str(response.metadata.get("error", "RPC server error."))
        )
    return response, int(request_wire_bytes), float(elapsed)


class _ThreadingRpcServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RpcRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        owner: "RpcServer" = self.server.owner
        packet_id = ""
        response_wire = 0
        try:
            request = receive_frame(
                self.request,
                max_message_bytes=owner.max_message_bytes,
            )
            response = owner.service.handle_rpc(request)
            packet_id = response.packet_id
            if float(response.delay_s) > 0.0:
                time.sleep(float(response.delay_s))
            response_wire = send_frame(
                self.request,
                response.message_type,
                response.metadata,
                response.payload,
                max_message_bytes=owner.max_message_bytes,
            )
            owner.service.record_response_wire(
                packet_id,
                int(response_wire),
            )
        except (RpcProtocolError, ValueError, KeyError) as exc:
            owner.service.record_transport_error(str(exc))
            try:
                response_wire = send_frame(
                    self.request,
                    "ERROR",
                    {"error": str(exc)},
                    max_message_bytes=owner.max_message_bytes,
                )
            except OSError:
                owner.service.record_transport_error(
                    "Peer closed before protocol error response."
                )
        except OSError as exc:
            owner.service.record_transport_error(
                f"Socket response failed: {exc}"
            )


class RpcServer:
    """Lifecycle wrapper for the localhost threaded RPC listener."""

    def __init__(
        self,
        service,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.service = service
        self.max_message_bytes = int(max_message_bytes)
        self._server = _ThreadingRpcServer(
            (str(host), int(port)),
            _RpcRequestHandler,
        )
        self._server.owner = self
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self._server.server_address
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RPC server is already running.")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fedagg-rpc-listener",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("RPC listener did not terminate.")
            self._thread = None
