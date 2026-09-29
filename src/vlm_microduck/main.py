"""Stream simulated camera frames into the red-box detector."""

from __future__ import annotations

import argparse
import math
import os
import signal
import time
import webbrowser
from pathlib import Path

from .camera import CameraError, SimulatorCamera
from .controller import AlignmentController, ControllerError
from .detector import annotate_red_box, find_red_box
from .preview import CameraPreview


POSITIONS = {
    "left": "左侧",
    "center": "中间",
    "right": "右侧",
    "not found": "未找到",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a red box in the MicroDuck MuJoCo camera.")
    parser.add_argument("--host", default=os.environ.get("DUCK_CAMERA_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("DUCK_CAMERA_PORT", "7901"))
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--preview-host", default=os.environ.get("DUCK_PREVIEW_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--preview-port", type=int, default=int(os.environ.get("DUCK_PREVIEW_PORT", "8081"))
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the preview page automatically"
    )
    parser.add_argument(
        "--align", action="store_true", help="turn the simulated duck toward the red target"
    )
    state_dir = Path(os.environ.get("DUCK_SIM_STATE", str(Path.home() / ".cache/duck-sim")))
    duck_name = os.environ.get("DUCK_SIM_DUCK", "duck-a")
    parser.add_argument(
        "--robot-socket",
        default=os.environ.get("DUCK_ROBOT_SOCKET", str(state_dir / f"{duck_name}.sock")),
        help="robotd Unix socket (used with --align)",
    )
    parser.add_argument(
        "--turn-speed", type=float, default=1.5,
        help="requested yaw rate in rad/s, up to 1.5 (not the measured body speed)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.turn_speed) or not 0 < args.turn_speed <= 1.5:
        parser.error("--turn-speed must be finite and between 0 and 1.5 rad/s")

    camera = SimulatorCamera(args.host, args.port, args.width, args.height)
    controller = (
        AlignmentController(args.robot_socket, turn_speed=args.turn_speed) if args.align else None
    )
    def change_control(action: str) -> None:
        if controller is not None:
            if action == "pause":
                controller.pause()
            else:
                controller.resume()

    preview = CameraPreview(
        args.preview_host,
        args.preview_port,
        control_status=(lambda: controller.status) if controller is not None else None,
        control_snapshot=controller.snapshot if controller is not None else None,
        control_action=change_control if controller is not None else None,
    )

    def interrupt_on_terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_on_terminate)
    try:
        preview.start()
        if controller is not None:
            controller.start()
        print(
            f"摄像头预览：{preview.url}\n"
            f"读取仿真摄像头 {args.host}:{args.port}。"
            f"{'自动对准已开启。' if controller else '仅预览。'}按 Ctrl+C 停止。",
            flush=True,
        )
        if controller is not None:
            print("手动接管会暂停自动对准；预览页可切换手动控制 / 恢复自动对准。", flush=True)
        if not args.no_browser:
            try:
                webbrowser.open(preview.url, new=2)
            except webbrowser.Error:
                pass
        while True:
            preview.set_status(f"正在连接仿真摄像头 {args.host}:{args.port}…")
            try:
                for frame in camera.frames():
                    detection = find_red_box(frame)
                    position = detection.position if detection else "not found"
                    if controller is not None:
                        controller.update(detection.position if detection else None)
                    detail = ""
                    if detection:
                        x, _, box_width, _ = detection.bounds
                        center_percent = round((x + box_width / 2) / frame.shape[1] * 100)
                        detail = f"（画面横向 {center_percent}%）"
                    preview.set_status(f"摄像头已连接 · 红方块：{POSITIONS[position]}{detail}")
                    preview.publish(annotate_red_box(frame, detection))
                    timestamp = time.strftime("%H:%M:%S")
                    action = f" · {controller.status}" if controller is not None else ""
                    print(
                        f"\r[{timestamp}] 红方块：{POSITIONS[position]}{detail}{action}\033[K",
                        end="",
                        flush=True,
                    )
            except (OSError, CameraError) as error:
                if controller is not None:
                    controller.stop("自动对准：摄像头断开，停止")
                preview.set_status(f"摄像头暂不可用，正在重试：{error}")
                print(f"\n摄像头暂不可用：{error}；2 秒后重试。", flush=True)
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n检测已停止。", flush=True)
    except ControllerError as error:
        print(f"\n自动对准已停止：{error}", flush=True)
        raise SystemExit(1) from error
    finally:
        if controller is not None:
            controller.close()
        preview.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
