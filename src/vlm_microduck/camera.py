"""Read the local MuJoCo simulator's raw camera frames."""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterator

import cv2
import numpy as np


class CameraError(Exception):
    """The simulator camera connection ended or sent an unexpected frame."""


class SimulatorCamera:
    """Connect to `duck-body`'s local frame port and yield upright RGB images.

    The simulator sends each frame as a four-byte little-endian length followed by
    640x360 UYVY pixels. The camera is mounted sideways, so frames are rotated upright
    here before the detector receives them.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7901,
        width: int = 640,
        height: int = 360,
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.timeout = timeout

    def frames(self) -> Iterator[np.ndarray]:
        """Yield RGB frames until the simulator disconnects or the caller stops."""
        expected_bytes = self.width * self.height * 2
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as stream:
            stream.settimeout(self.timeout)
            while True:
                length = struct.unpack("<I", self._read_exact(stream, 4))[0]
                if length != expected_bytes:
                    raise CameraError(
                        f"expected a {expected_bytes}-byte UYVY frame, received {length} bytes"
                    )
                packed = np.frombuffer(self._read_exact(stream, length), dtype=np.uint8)
                uyvy = packed.reshape((self.height, self.width, 2))
                rgb_sideways = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
                yield cv2.rotate(rgb_sideways, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def _read_exact(stream: socket.socket, size: int) -> bytes:
        data = bytearray(size)
        view = memoryview(data)
        offset = 0
        while offset < size:
            received = stream.recv_into(view[offset:])
            if received == 0:
                raise CameraError("the simulator closed the camera connection")
            offset += received
        return bytes(data)
