"""Exercise control behavior through the actual newline-delimited Unix socket wire."""

from __future__ import annotations

import contextlib
import io
import json
import socket
import socketserver
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vlm_microduck.camera import CameraError
from vlm_microduck.controller import AlignmentController, ControllerError
from vlm_microduck.main import main


class FakeRobot:
    def __init__(self, healthy: bool = True, sitting: bool = False) -> None:
        self.healthy = healthy
        self.sitting = sitting
        self.twist = [0.0, 0.0, 0.0]
        self.stream_states = True
        self.finished = threading.Event()
        self.moves: list[dict] = []
        self.connections: list[socket.socket] = []
        self.condition = threading.Condition()
        self.directory = tempfile.TemporaryDirectory(prefix="vlm-control-")
        self.path = Path(self.directory.name) / "robot.sock"
        robot = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                with robot.condition:
                    robot.connections.append(self.request)
                try:
                    for line in self.rfile:
                        message = json.loads(line)
                        method = message["method"]
                        if method == "robot.move":
                            with robot.condition:
                                robot.moves.append(message["params"])
                                robot.twist = [message["params"][key] for key in ("vx", "vy", "vyaw")]
                                robot.condition.notify_all()
                            if "id" in message:
                                self.reply(message, {"accepted": True})
                        elif method == "robot.health":
                            self.reply(message, {"healthy": robot.healthy, "reason": "test"})
                        elif method == "robot.policies":
                            self.reply(message, {
                                "enabled": True,
                                "homed": True,
                                "sitting": robot.sitting,
                                "slots": [{"slot": "walk", "path": "walking.onnx"}],
                            })
                        elif method == "robot.subscribe":
                            self.reply(message, {"accepted": True})
                            while not robot.finished.is_set():
                                if robot.stream_states:
                                    with robot.condition:
                                        state = {
                                            "move": {"requested": list(robot.twist)},
                                            "safety": {"fallen": False, "limp": False},
                                            "imu": {"gyro": [0, 0, 0.12]},
                                        }
                                    self.wfile.write((json.dumps({
                                        "jsonrpc": "2.0", "method": "robot.state", "params": state
                                    }) + "\n").encode("utf-8"))
                                    self.wfile.flush()
                                robot.finished.wait(0.02)
                            return
                except OSError:
                    return

            def reply(self, request: dict, result: dict) -> None:
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
                self.wfile.flush()

        self.server = socketserver.ThreadingUnixStreamServer(str(self.path), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def wait_moves(self, count: int) -> list[dict]:
        with self.condition:
            if not self.condition.wait_for(lambda: len(self.moves) >= count, timeout=2):
                raise AssertionError(f"expected {count} moves, received {self.moves}")
            return list(self.moves)

    def disconnect(self) -> None:
        for connection in self.connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)

    def close(self) -> None:
        self.finished.set()
        self.disconnect()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def drive(self, vx=0.0, vyaw=0.0) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.path))
            connection.sendall((json.dumps({
                "jsonrpc": "2.0", "method": "robot.move",
                "params": {"vx": vx, "vy": 0.0, "vyaw": vyaw},
            }) + "\n").encode("utf-8"))


def wait_until(predicate, timeout=2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = FakeRobot()
        self.controller = AlignmentController(self.robot.path, turn_speed=0.25)
        self.controller.start()
        self.addCleanup(self.robot.close)
        self.addCleanup(self.controller.close)

    def test_turn_directions_and_confirmed_center_or_lost_stop(self) -> None:
        for position in (
            "left", "left", "left", "center", "center", "center",
            "right", "right", "right", None,
        ):
            self.controller.update(position)
        moves = self.robot.wait_moves(4)
        self.assertEqual(
            [move["vyaw"] for move in moves[:4]],
            [0.25, 0, -0.25, 0],
        )
        # A direction bug must never be hidden by also translating the duck.
        self.assertTrue(all(move["vx"] == 0 and move["vy"] == 0 for move in moves))

    def test_single_center_or_occluded_frame_does_not_abort_the_turn(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.robot.wait_moves(1)
        self.controller.update("center")
        self.controller.update("left")
        self.controller.update(None)
        self.controller.update("left")
        self.assertTrue(self.controller.snapshot()["enabled"])
        self.assertEqual(len(self.robot.moves), 1)
        self.assertIn("请求左转", self.controller.status)

    def test_missing_target_grace_expires_even_without_another_frame(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.controller.update(None)
        moves = self.robot.wait_moves(2)
        self.assertEqual(moves[1]["vyaw"], 0)
        self.assertIn("目标丢失", self.controller.status)

    def test_one_opposite_detection_stops_instead_of_reversing(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.controller.update("right")
        moves = self.robot.wait_moves(2)
        self.assertEqual(moves[0]["vyaw"], 0.25)
        self.assertEqual(moves[1]["vyaw"], 0)

    def test_stalled_camera_stops_without_another_update(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        moves = self.robot.wait_moves(2)
        self.assertEqual(moves[0]["vyaw"], 0.25)
        self.assertEqual(moves[1]["vyaw"], 0)
        self.assertIn("画面超时", self.controller.status)
        # After stale input, one fresh frame cannot immediately resume turning.
        self.controller.update("left")
        self.assertEqual(len(self.robot.moves), 2)

    def test_close_explicitly_sends_zero_velocity(self) -> None:
        for _ in range(3):
            self.controller.update("right")
        self.controller.close()
        moves = self.robot.wait_moves(2)
        self.assertEqual(moves[0]["vyaw"], -0.25)
        self.assertEqual(moves[1], {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
        self.controller.close()  # Cleanup is safe to repeat.

    def test_frame_gap_resets_confirmation_even_before_watchdog_runs(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        with self.controller._lock:
            self.controller._last_frame = time.monotonic() - 1
        self.controller.update("left")
        self.assertEqual(self.robot.wait_moves(2)[1]["vyaw"], 0)

    def test_broken_control_connection_is_reported(self) -> None:
        self.robot.disconnect()
        wait_until(lambda: self.controller.error is not None)
        with self.assertRaises(ControllerError):
            self.controller.update("left")
        self.assertIn("连接中断", self.controller.error or "")
        with self.assertRaises(ControllerError):
            self.controller.update("left")

    def test_idle_detections_do_not_overwrite_other_drivers(self) -> None:
        for _ in range(20):
            self.controller.update("center")
            self.controller.update(None)
        time.sleep(0.05)
        self.assertEqual(self.robot.moves, [])

    def test_pause_leaves_manual_forward_and_turn_untouched_even_on_exit(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.controller.pause()
        self.robot.drive(vx=0.15, vyaw=-0.4)
        self.robot.wait_moves(3)
        for _ in range(30):
            self.controller.update("left")
            self.controller.update(None)
        time.sleep(0.35)  # Watchdog must not zero a manual intent either.
        self.controller.stop("camera disconnected")
        self.controller.close()
        self.assertEqual(len(self.robot.moves), 3)
        self.assertEqual(self.robot.moves[-1], {"vx": 0.15, "vy": 0.0, "vyaw": -0.4})

    def test_external_manual_intent_pauses_without_sending_zero(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.robot.wait_moves(1)
        time.sleep(0.1)  # Drain old states from before the automatic turn.
        self.robot.drive(vx=0.15, vyaw=-0.4)
        wait_until(lambda: not self.controller.snapshot()["enabled"])
        self.assertIn("手动接管", self.controller.status)
        for _ in range(10):
            self.controller.update("left")
        self.controller.close()
        self.assertEqual(len(self.robot.moves), 2)
        self.assertEqual(self.robot.moves[-1]["vx"], 0.15)

    def test_resume_requires_new_confirmation_then_turns_again(self) -> None:
        self.controller.pause()
        self.controller.update("left")
        self.controller.resume()
        self.robot.wait_moves(1)  # Explicit resume sends one zero.
        self.controller.update("right")
        self.controller.update("right")
        self.assertEqual(len(self.robot.moves), 1)
        self.controller.update("right")
        self.assertEqual(self.robot.wait_moves(2)[1]["vyaw"], -0.25)

    def test_state_stream_timeout_stops_owned_motion(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.robot.stream_states = False
        wait_until(lambda: self.controller.error is not None)
        self.assertEqual(self.robot.wait_moves(2)[1]["vyaw"], 0)
        self.assertIn("状态超时", self.controller.error)

    def test_requested_motion_is_throttled_and_actual_motion_is_reported(self) -> None:
        for _ in range(100):
            self.controller.update("left")
        self.robot.wait_moves(1)
        self.assertEqual(len(self.robot.moves), 1)
        wait_until(lambda: self.controller.snapshot()["measured_yaw_rate"] is not None)
        self.assertAlmostEqual(self.controller.snapshot()["measured_yaw_rate"], 0.12)

    def test_a_turn_that_is_not_physically_happening_is_paused(self) -> None:
        for _ in range(3):
            self.controller.update("left")
        self.robot.wait_moves(1)
        with self.controller._lock:
            self.controller._turn_started = time.monotonic() - 3
            self.controller._rates.clear()
            self.controller._rates.extend([0.0] * 50)
        wait_until(lambda: not self.controller.snapshot()["enabled"])
        self.assertEqual(self.robot.wait_moves(2)[1]["vyaw"], 0)
        self.assertIn("未检测到持续转向", self.controller.status)


class ReadinessTests(unittest.TestCase):
    def test_sitting_robot_is_rejected_without_movement(self) -> None:
        robot = FakeRobot(sitting=True)
        controller = AlignmentController(robot.path)
        try:
            with self.assertRaisesRegex(ControllerError, "尚未站起"):
                controller.start()
            self.assertEqual(robot.moves, [])
        finally:
            controller.close()
            robot.close()

    def test_already_moving_robot_keeps_manual_control_on_start(self) -> None:
        robot = FakeRobot()
        robot.twist = [0.1, 0, -0.4]
        controller = AlignmentController(robot.path)
        try:
            controller.start()
            for _ in range(10):
                controller.update("left")
            self.assertFalse(controller.snapshot()["enabled"])
            self.assertEqual(robot.moves, [])
        finally:
            controller.close()
            robot.close()
    def test_unhealthy_robot_receives_no_movement(self) -> None:
        robot = FakeRobot(healthy=False)
        controller = AlignmentController(robot.path)
        try:
            with self.assertRaisesRegex(ControllerError, "尚未就绪"):
                controller.start()
            self.assertEqual(robot.moves, [])
        finally:
            controller.close()
            robot.close()

    def test_nonfinite_or_out_of_range_speed_is_rejected(self) -> None:
        for speed in (float("nan"), float("inf"), 0, -0.25, 1.6):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                AlignmentController("/unused.sock", turn_speed=speed)

    def test_default_speed_matches_the_cpu_rehearsal_turn_limit(self) -> None:
        self.assertEqual(AlignmentController("/unused.sock").turn_speed, 1.5)

    def test_missing_socket_has_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vlm-missing-") as directory:
            controller = AlignmentController(Path(directory) / "missing.sock")
            with self.assertRaises(ControllerError):
                controller.start()
            controller.close()


class MainLoopTests(unittest.TestCase):
    def test_camera_disconnect_and_exit_both_send_stop(self) -> None:
        robot = FakeRobot()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[45:100, 10:40] = (255, 0, 0)

        class DisconnectingCamera:
            def frames(self):
                for _ in range(3):
                    yield frame
                raise CameraError("test camera disconnected")

        argv = [
            "vlm-microduck", "--align", "--no-browser", "--preview-port", "0",
            "--robot-socket", str(robot.path),
            "--turn-speed", "0.25",
        ]
        try:
            with (
                patch("sys.argv", argv),
                patch("vlm_microduck.main.SimulatorCamera", return_value=DisconnectingCamera()),
                patch("vlm_microduck.main.time.sleep", side_effect=KeyboardInterrupt),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            moves = robot.wait_moves(2)
            self.assertEqual(moves[0]["vyaw"], 0.25)
            self.assertEqual(moves[1]["vyaw"], 0)  # Camera exception.
            self.assertEqual(len(moves), 2)  # No redundant zero on exit.
        finally:
            robot.close()


if __name__ == "__main__":
    unittest.main()
