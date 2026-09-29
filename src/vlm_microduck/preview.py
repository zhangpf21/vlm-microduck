"""Serve annotated camera frames as a small local browser preview."""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Thread

import cv2
import numpy as np


_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MicroDuck 摄像头预览</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #101318; color: #eef2f7; }
    main { width: min(1100px, calc(100% - 32px)); margin: 24px 0; }
    header { display: flex; align-items: baseline; justify-content: space-between;
      gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
    h1 { margin: 0; font-size: 1.25rem; }
    #status { color: #aab6c5; font-size: .95rem; }
    .frame { min-height: 240px; display: grid; place-items: center;
      overflow: hidden; border: 1px solid #303844; border-radius: 12px;
      background: #050608; }
    img { display: block; max-width: 100%; max-height: 78vh; object-fit: contain; }
    footer { color: #8491a1; font-size: .85rem; margin-top: 12px; }
    .controls { margin: 12px 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .controls[hidden] { display: none; }
    button { color: #eef2f7; background: #28384e; border: 1px solid #536780;
      border-radius: 6px; padding: 8px 12px; cursor: pointer; }
    button:disabled { opacity: .45; cursor: default; }
    #motion { color: #aab6c5; font-size: .9rem; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MicroDuck · 红色目标检测</h1>
      <div id="status">等待摄像头连接…</div>
    </header>
    <div class="frame"><img src="/stream" alt="实时摄像头画面"></div>
    <div class="controls" id="controls" hidden>
      <button id="pause" type="button">切换手动控制</button>
      <button id="resume" type="button">恢复自动对准</button>
      <span id="motion"></span>
      <span>手动驾驶请使用 <a href="http://127.0.0.1:8080" target="_blank" rel="noopener">机器人控制台</a>。</span>
    </div>
    <footer>两条竖线将画面分为左、中、右区域。关闭此页不会停止检测；在运行终端按 Ctrl+C 停止。</footer>
  </main>
  <script>
    async function refreshStatus() {
      try {
        const response = await fetch('/status', { cache: 'no-store' });
        const value = await response.json();
        document.querySelector('#status').textContent = value.status;
        const control = value.control;
        document.querySelector('#controls').hidden = !control;
        if (control) {
          document.querySelector('#pause').disabled = !control.enabled || !!control.error;
          document.querySelector('#resume').disabled = control.enabled || !!control.error;
          const rate = control.measured_yaw_rate;
          document.querySelector('#motion').textContent = rate == null
            ? '等待实测运动状态'
            : '实测转向角速度（1 秒均值）：' + rate.toFixed(3) + ' rad/s';
        }
      } catch (_) {
        document.querySelector('#status').textContent = '预览服务连接中断';
      }
    }
    async function changeControl(action) {
      try {
        const response = await fetch('/control', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action })
        });
        const value = await response.json();
        if (!response.ok) throw new Error(value.error || '切换失败');
        await refreshStatus();
      } catch (error) {
        document.querySelector('#status').textContent = error.message;
      }
    }
    document.querySelector('#pause').onclick = () => changeControl('pause');
    document.querySelector('#resume').onclick = () => changeControl('resume');
    refreshStatus();
    setInterval(refreshStatus, 500);
  </script>
</body>
</html>"""


class CameraPreview:
    """Publish RGB frames to a browser using an MJPEG stream."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8081,
        control_status: Callable[[], str] | None = None,
        control_snapshot: Callable[[], dict] | None = None,
        control_action: Callable[[str], None] | None = None,
    ) -> None:
        self._condition = Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._status = "等待仿真摄像头画面…"
        self._closed = False
        self._control_status = control_status
        self._control_snapshot = control_snapshot
        self._control_action = control_action
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread = Thread(target=self._server.serve_forever, name="camera-preview", daemon=True)
        bound_host, bound_port = self._server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
        self.url = f"http://{display_host}:{bound_port}/"

    def start(self) -> None:
        self._thread.start()

    def set_status(self, status: str) -> None:
        with self._condition:
            self._status = status
            self._condition.notify_all()

    def publish(self, rgb_frame: np.ndarray) -> None:
        """Encode and publish one annotated RGB frame."""
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("OpenCV could not encode the camera preview frame")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=2)
        self._server.server_close()

    def _current_status(self) -> str:
        with self._condition:
            status = self._status
        if self._control_status is not None:
            status += " · " + self._control_status()
        return status

    def _next_frame(self, after_sequence: int) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._closed,
                timeout=1.0,
            )
            if self._sequence > after_sequence and self._jpeg is not None:
                return self._sequence, self._jpeg
            return None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, value: dict, status: int = 200) -> None:
                body = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/control" or preview._control_action is None:
                    self.send_error(404)
                    return
                # No CORS support. Require JSON and same-origin browser requests
                # so an unrelated website cannot take over local robot control.
                origin = self.headers.get("Origin")
                if (self.headers.get("Content-Type") != "application/json"
                        or origin and origin != "http://" + self.headers.get("Host", "")):
                    self._json({"error": "只接受同源 JSON 控制请求"}, 403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("无效的请求长度")
                    value = json.loads(self.rfile.read(length))
                    action = value["action"]
                    if action not in ("pause", "resume"):
                        raise ValueError("只支持 pause / resume")
                except (ValueError, KeyError, TypeError, UnicodeError):
                    self._json({"error": "无效的控制请求"}, 400)
                    return
                try:
                    preview._control_action(action)
                except Exception as error:
                    self._json({"error": str(error)}, 409)
                    return
                self._json({"control": preview._control_snapshot()})

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path == "/" or self.path.startswith("/?"):
                    body = _PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/status":
                    self._json({
                        "status": preview._current_status(),
                        "control": preview._control_snapshot()
                        if preview._control_snapshot is not None else None,
                    })
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    sequence = 0
                    try:
                        while not preview._closed:
                            item = preview._next_frame(sequence)
                            if item is None:
                                continue
                            sequence, jpeg = item
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self.send_error(404)

            def log_message(self, format: str, *args: object) -> None:
                # Browser status polling happens once per second; keep the terminal readable.
                return

        return Handler
