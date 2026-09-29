"""Check the pause/resume browser endpoints without moving a real robot."""

import http.client
import json
import unittest
from urllib.parse import urlsplit

from vlm_microduck.preview import CameraPreview


class PreviewControlTests(unittest.TestCase):
    def setUp(self):
        self.enabled = True
        self.actions = []

        def action(value):
            self.actions.append(value)
            self.enabled = value == "resume"

        self.preview = CameraPreview(
            port=0,
            control_status=lambda: "自动对准" if self.enabled else "手动控制",
            control_snapshot=lambda: {
                "enabled": self.enabled, "status": "test",
                "measured_yaw_rate": 0.12, "error": None,
            },
            control_action=action,
        )
        self.preview.start()
        self.addCleanup(self.preview.close)
        self.address = urlsplit(self.preview.url)

    def request(self, method, path, value=None, headers=None):
        connection = http.client.HTTPConnection(self.address.hostname, self.address.port, timeout=2)
        try:
            connection.request(
                method, path, body=json.dumps(value) if value is not None else None,
                headers=headers or {},
            )
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_status_includes_mode_and_measured_not_requested_speed(self):
        status, body = self.request("GET", "/status")
        self.assertEqual(status, 200)
        value = json.loads(body)
        self.assertTrue(value["control"]["enabled"])
        self.assertEqual(value["control"]["measured_yaw_rate"], 0.12)

    def test_page_has_manual_and_resume_buttons(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("切换手动控制".encode(), body)
        self.assertIn("恢复自动对准".encode(), body)

    def test_pause_and_resume_call_the_controller(self):
        for action in ("pause", "resume"):
            status, body = self.request(
                "POST", "/control", {"action": action},
                {"Content-Type": "application/json", "Origin": self.preview.url.rstrip("/")},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["control"]["enabled"], action == "resume")
        self.assertEqual(self.actions, ["pause", "resume"])

    def test_cross_origin_and_form_requests_cannot_change_mode(self):
        for headers in (
            {"Content-Type": "application/json", "Origin": "https://unrelated.example"},
            {"Content-Type": "application/x-www-form-urlencoded"},
        ):
            status, _ = self.request("POST", "/control", {"action": "pause"}, headers)
            self.assertEqual(status, 403)
        self.assertEqual(self.actions, [])

    def test_invalid_action_does_not_call_controller(self):
        status, _ = self.request(
            "POST", "/control", {"action": "drive"}, {"Content-Type": "application/json"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(self.actions, [])

    def test_preview_only_mode_has_no_control_endpoint(self):
        preview = CameraPreview(port=0)
        preview.start()
        self.addCleanup(preview.close)
        address = urlsplit(preview.url)
        connection = http.client.HTTPConnection(address.hostname, address.port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("POST", "/control")
        self.assertEqual(connection.getresponse().status, 404)
