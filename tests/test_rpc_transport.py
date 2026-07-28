import socket
import threading

import pytest

from rpc_transport import (
    RpcResponse,
    RpcServer,
    RpcProtocolError,
    encode_frame,
    receive_frame,
)


class _NoTaskService:
    def handle_rpc(self, request):
        return RpcResponse("NO_TASK", {"method": request.message_type})

    def record_response_wire(self, packet_id, response_wire_bytes):
        assert response_wire_bytes > 0

    def record_transport_error(self, message):
        raise AssertionError(message)


def test_rpc_framing_handles_partial_tcp_receive():
    left, right = socket.socketpair()
    frame = encode_frame(
        "UPLOAD_KNOWLEDGE",
        {"client_id": 2},
        b"real-binary-payload",
    )

    def writer():
        for offset in range(0, len(frame), 3):
            left.sendall(frame[offset:offset + 3])
        left.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        decoded = receive_frame(right)
    finally:
        right.close()
        thread.join(timeout=2.0)

    assert decoded.message_type == "UPLOAD_KNOWLEDGE"
    assert decoded.metadata["client_id"] == 2
    assert decoded.payload == b"real-binary-payload"


def test_rpc_framing_rejects_oversized_message_before_allocation():
    with pytest.raises(RpcProtocolError, match="max_message_bytes"):
        encode_frame(
            "UPLOAD_KNOWLEDGE",
            {},
            b"x" * 32,
            max_message_bytes=8,
        )


def test_rpc_server_clean_shutdown_releases_its_tcp_port():
    server = RpcServer(_NoTaskService())
    server.start()
    host, port = server.address
    server.close()

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    finally:
        probe.close()
