"""Rotation-only vision control with an explicit hand-off to manual driving."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import BinaryIO


class ControllerError(Exception):
    """The controller cannot establish or maintain its robot connection."""


class AlignmentController:
    """Only renew an owned motion on fresh frames; yield to other drivers.

    robot.move is last-writer-wins, not a queue or a per-client lease. Monitor
    robot.state for a different intent and latch into manual mode, without
    writing a stop over that driver's command. Explicit pause/resume is the
    reliable hand-off when another client sends the same yaw as this one.
    """

    def __init__(
        self,
        socket_path: str | Path,
        turn_speed: float = 1.5,
        frame_timeout: float = 0.3,
        confirm_frames: int = 3,
    ) -> None:
        if not math.isfinite(turn_speed) or not 0 < turn_speed <= 1.5:
            raise ValueError("turn_speed must be finite and between 0 and 1.5 rad/s")
        if not math.isfinite(frame_timeout) or frame_timeout <= 0:
            raise ValueError("frame_timeout must be finite and positive")
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least one")
        self.socket_path = str(Path(socket_path).expanduser())
        self.turn_speed = turn_speed
        self.frame_timeout = frame_timeout
        self.confirm_frames = confirm_frames
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._threads: list[threading.Thread] = []
        self._socket: socket.socket | None = None
        self._observer: socket.socket | None = None
        self._enabled = True
        self._owns_motion = False
        self._expected_yaw = 0.0
        self._last_sent = 0.0
        self._echo_grace_until = 0.0
        self._turn_started: float | None = None
        self._last_frame: float | None = None
        self._last_seen: float | None = None
        self._target_missing = False
        self._center_frames = 0
        self._last_state: float | None = None
        self._candidate: str | None = None
        self._confirmed = 0
        self._rates: deque[float] = deque(maxlen=50)
        self._status = "自动对准：等待摄像头"
        self._error: str | None = None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled and self._error is None,
                "status": self._status,
                "measured_yaw_rate": sum(self._rates) / len(self._rates) if self._rates else None,
                "error": self._error,
            }

    def start(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(1.0)
        observer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        observer.settimeout(1.0)
        reader = None
        try:
            connection.connect(self.socket_path)
            with connection.makefile("rb") as replies:
                health = self._request(connection, replies, "robot.health", 1)
                if not health.get("healthy"):
                    raise ControllerError(f"robotd 尚未就绪：{health.get('reason') or 'unhealthy'}")
                policies = self._request(connection, replies, "robot.policies", 2)
                walking = any(
                    item.get("slot") == "walk" and item.get("path")
                    for item in policies.get("slots", [])
                )
                if not policies.get("enabled") or not walking:
                    raise ControllerError("未加载运动策略，请先启动 scripts/duck-sim")
                if policies.get("homed") is False or policies.get("sitting") is True:
                    raise ControllerError("Duck 尚未站起，请等待仿真完成站立后重试")
            observer.connect(self.socket_path)
            reader = observer.makefile("rb")
            subscribed = self._request(observer, reader, "robot.subscribe", 3, {"hz": 50})
            if not subscribed.get("accepted"):
                raise ControllerError("robotd 拒绝了运动状态订阅")
            # Observe an existing manual intent BEFORE issuing any movement.
            state = self._read_state(reader)
            connection.settimeout(0.2)
            observer.settimeout(None)
            with self._lock:
                self._socket = connection
                self._observer = observer
                self._status = "自动对准：等待目标"
                self._observe_locked(state)
        except (OSError, ControllerError, ValueError) as error:
            if reader is not None:
                reader.close()
            observer.close()
            connection.close()
            self._socket = None
            self._observer = None
            raise ControllerError(f"{self.socket_path}：{error}") from error
        self._threads = [
            threading.Thread(target=self._watch_frames, name="alignment-watchdog", daemon=True),
            threading.Thread(
                target=self._watch_robot, args=(reader,), name="alignment-state", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _request(
        connection: socket.socket, reader: BinaryIO, method: str, request_id: int,
        params: dict | None = None,
    ) -> dict:
        request = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        line = reader.readline(65536)
        if not line:
            raise ControllerError("robotd 关闭了连接")
        try:
            response = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise ControllerError("robotd 返回了无效的 JSON") from error
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise ControllerError(f"{method} 的响应编号不匹配")
        if "error" in response:
            raise ControllerError(f"{method} 被拒绝：{response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ControllerError(f"{method} 缺少有效结果")
        return result

    @staticmethod
    def _read_state(reader: BinaryIO) -> dict:
        line = reader.readline(65536)
        if not line:
            raise ControllerError("运动状态连接中断")
        try:
            message = json.loads(line)
            if message["method"] != "robot.state" or not isinstance(message["params"], dict):
                raise ValueError("not a robot.state notification")
            return message["params"]
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            raise ControllerError("robotd 返回了无效的运动状态") from error

    def _reset_detection_locked(self) -> None:
        self._last_frame = None
        self._last_seen = None
        self._target_missing = False
        self._center_frames = 0
        self._candidate = None
        self._confirmed = 0

    def pause(self) -> None:
        with self._lock:
            self._check_error_locked()
            self._send_locked(
                0.0, "手动模式：自动对准已暂停，可使用控制台驾驶", acknowledge=True
            )
            self._enabled = False
            self._reset_detection_locked()

    def resume(self) -> None:
        with self._lock:
            self._check_error_locked()
            if self._last_state is None or time.monotonic() - self._last_state > 0.5:
                raise ControllerError("机器人状态已超时，不能恢复自动对准")
            self._enabled = True
            self._reset_detection_locked()
            # Clicking resume explicitly transfers control back from the user.
            self._send_locked(
                0.0, "自动对准：已恢复，等待目标", force=True, acknowledge=True
            )

    def update(self, position: str | None) -> None:
        if position not in ("left", "center", "right", None):
            raise ValueError("position must be left, center, right, or None")
        with self._lock:
            self._check_error_locked()
            if not self._enabled:
                return
            now = time.monotonic()
            if self._last_frame is not None and now - self._last_frame >= self.frame_timeout:
                self._candidate = None
                self._confirmed = 0
                self._last_seen = None
                self._center_frames = 0
            self._last_frame = now
            if position is None:
                self._center_frames = 0
                self._target_missing = True
                if (
                    self._owns_motion and self._last_seen is not None
                    and now - self._last_seen < 0.1
                ):
                    # One occluded frame during a footstep should not abort the
                    # gait. The watchdog bounds this grace even if frames stall.
                    self._send_locked(self._expected_yaw, "自动对准：目标短暂遮挡，确认中")
                    return
                self._candidate = None
                self._confirmed = 0
                self._send_locked(0.0, "自动对准：目标丢失，停止")
                return
            self._last_seen = now
            self._target_missing = False
            if position == "center":
                self._center_frames += 1
                if self._owns_motion and self._center_frames < self.confirm_frames:
                    self._send_locked(self._expected_yaw, "自动对准：确认目标居中")
                    return
                self._candidate = None
                self._confirmed = 0
                self._send_locked(0.0, "自动对准：目标居中，停止")
                return
            self._center_frames = 0
            if position == self._candidate:
                self._confirmed += 1
            else:
                self._candidate = position
                self._confirmed = 1
            if self._confirmed < self.confirm_frames:
                self._send_locked(0.0, "自动对准：确认目标位置，停止")
                return
            yaw = self.turn_speed if position == "left" else -self.turn_speed
            direction = "左转" if yaw > 0 else "右转"
            self._send_locked(yaw, f"自动对准：请求{direction} {self.turn_speed:g} rad/s")

    def stop(self, reason: str = "自动对准：已停止") -> None:
        with self._lock:
            self._reset_detection_locked()
            if self._enabled:
                self._send_locked(0.0, reason)

    def _check_error_locked(self) -> None:
        if self._error:
            raise ControllerError(self._error)
        if self._socket is None:
            raise ControllerError("控制连接尚未建立")

    def _send_locked(
        self, yaw: float, status: str, force: bool = False, acknowledge: bool = False
    ) -> None:
        self._check_error_locked()
        self._status = status
        now = time.monotonic()
        # Do not keep writing idle zeros over the manual driver's commands.
        if not force and yaw == 0.0 and not self._owns_motion:
            return
        if not force and yaw != 0.0 and yaw == self._expected_yaw and now - self._last_sent < 0.1:
            return
        notification = {
            "jsonrpc": "2.0", "method": "robot.move",
            "params": {"vx": 0.0, "vy": 0.0, "vyaw": yaw},
        }
        try:
            if acknowledge:
                # A hand-off must finish applying the stop before a new driver
                # writes on another socket; send-order alone cannot guarantee it.
                with self._socket.makefile("rb") as replies:
                    result = self._request(
                        self._socket, replies, "robot.move", 4, notification["params"]
                    )
                if not result.get("accepted"):
                    raise ControllerError("robotd 未确认停止指令")
            else:
                self._socket.sendall((json.dumps(notification) + "\n").encode("utf-8"))
        except (OSError, ControllerError) as error:
            self._error = f"控制连接中断：{error}"
            self._status = self._error
            raise ControllerError(self._error) from error
        now = time.monotonic()
        if force or yaw != self._expected_yaw:
            # Allow old telemetry already in flight to drain, not an endless
            # grace period restarted by every heartbeat.
            self._echo_grace_until = now + 0.08
            self._turn_started = now if yaw else None
            if yaw:
                self._rates.clear()
        self._last_sent = now
        self._expected_yaw = yaw
        self._owns_motion = yaw != 0.0

    def _observe_locked(self, state: dict) -> None:
        try:
            requested = state["move"]["requested"]
            if len(requested) != 3 or not all(math.isfinite(float(v)) for v in requested):
                raise ValueError("invalid twist")
            safety = state["safety"]
        except (KeyError, TypeError, ValueError) as error:
            raise ControllerError("robot.state 缺少有效的运动/安全状态") from error
        now = time.monotonic()
        self._last_state = now
        gyro = state.get("imu", {}).get("gyro")
        if gyro is not None and len(gyro) == 3 and math.isfinite(gyro[2]):
            self._rates.append(float(gyro[2]))
        if not self._enabled:
            return
        if safety.get("fallen") or safety.get("limp"):
            self._send_locked(0.0, "自动对准已暂停：机器人倒地或处于松弛状态")
            self._enabled = False
            self._reset_detection_locked()
        elif now >= self._echo_grace_until and any(
            not math.isclose(float(actual), expected, abs_tol=1e-6)
            for actual, expected in zip(requested, (0.0, 0.0, self._expected_yaw))
        ):
            # Do NOT send zero here: the different intent belongs to the user.
            self._enabled = False
            self._owns_motion = False
            self._reset_detection_locked()
            self._status = "手动接管：自动对准已暂停；点“恢复自动对准”可重新开启"

    def _watch_robot(self, reader: BinaryIO) -> None:
        try:
            while not self._finished.is_set():
                state = self._read_state(reader)
                with self._lock:
                    self._observe_locked(state)
        except (OSError, ControllerError) as error:
            if not self._finished.is_set():
                with self._lock:
                    try:
                        if self._enabled:
                            self._send_locked(0.0, "自动对准：状态连接中断，停止")
                    except ControllerError:
                        pass
                    self._error = str(error)
                    self._status = self._error
        finally:
            reader.close()

    def _watch_frames(self) -> None:
        while not self._finished.wait(min(0.05, self.frame_timeout / 2)):
            with self._lock:
                now = time.monotonic()
                if self._last_state is not None and now - self._last_state > 0.5:
                    try:
                        if self._enabled:
                            self._send_locked(0.0, "自动对准：机器人状态超时，停止")
                    except ControllerError:
                        pass
                    self._error = "机器人状态超时，自动对准已停止"
                    self._status = self._error
                    return
                if not self._enabled or self._last_frame is None:
                    continue
                if (
                    self._owns_motion and self._target_missing and self._last_seen is not None
                    and now - self._last_seen >= 0.1
                ):
                    self._reset_detection_locked()
                    try:
                        self._send_locked(0.0, "自动对准：目标丢失，停止")
                    except ControllerError:
                        return
                    continue
                if now - self._last_frame >= self.frame_timeout:
                    self._reset_detection_locked()
                    try:
                        self._send_locked(0.0, "自动对准：画面超时，停止")
                    except ControllerError:
                        return
                elif (
                    self._owns_motion and self._turn_started is not None
                    and now - self._turn_started > 2.0 and len(self._rates) >= 25
                    and (sum(self._rates) / len(self._rates)) * self._expected_yaw
                    < 0.025 * abs(self._expected_yaw)
                ):
                    try:
                        self._send_locked(
                            0.0, "自动对准已暂停：未检测到持续转向，请检查步态或障碍物"
                        )
                    except ControllerError:
                        return
                    self._enabled = False
                    self._reset_detection_locked()

    def close(self) -> None:
        """Stop only our own rotation; never cancel the user's manual motion."""
        self._finished.set()
        if self._observer is not None:
            try:
                self._observer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for thread in self._threads:
            thread.join(timeout=1)
        with self._lock:
            if self._socket is not None:
                try:
                    if self._owns_motion:
                        # A state-stream failure may set _error while the command
                        # socket still works. Cleanup must still attempt a stop.
                        self._error = None
                        self._send_locked(0.0, "自动对准：已停止")
                except ControllerError:
                    pass  # robotd's deadman also expires a disconnected intent.
                finally:
                    self._socket.close()
                    self._socket = None
            if self._observer is not None:
                self._observer.close()
                self._observer = None
