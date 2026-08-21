import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
AUDIOCPP_SERVER_SHA256 = "9e4a0447c57a387f4626c9df2a3205c183d9df88fe8ddf1c79bc44a609ac7827"


class LauncherContractTests(unittest.TestCase):
    def test_macos_server_is_bundled_and_verified(self):
        server = ROOT / "app" / "audiocpp_server"
        license_file = ROOT / "app" / "audiocpp-LICENSE.txt"
        self.assertTrue(server.is_file())
        self.assertTrue(license_file.is_file())

        server_bytes = server.read_bytes()
        self.assertEqual(server_bytes[:8], bytes.fromhex("cffaedfe0c000001"))
        self.assertEqual(hashlib.sha256(server_bytes).hexdigest(), AUDIOCPP_SERVER_SHA256)
        self.assertIn(b"/v1/tasks/run-stream", server_bytes)
        self.assertIn(b"text/event-stream", server_bytes)

        install = (ROOT / "install.js").read_text(encoding="utf-8")
        self.assertIn(AUDIOCPP_SERVER_SHA256, install)
        self.assertNotIn("raw.githubusercontent.com/cocktailpeanut/minimax-music", install)

    def test_reset_keeps_bundled_server(self):
        reset = (ROOT / "reset.js").read_text(encoding="utf-8")
        self.assertNotIn("app/audiocpp_server", reset)

    def test_macos_selects_fast_config_only_with_enough_memory(self):
        start = (ROOT / "start.js").read_text(encoding="utf-8")
        fast_config = (ROOT / "app" / "audio.cpp-server-fast.json").read_text(encoding="utf-8")
        low_memory_config = (ROOT / "app" / "audio.cpp-server.json").read_text(encoding="utf-8")
        self.assertIn("os.totalmem() >= 32000000000", start)
        self.assertIn("audio.cpp-server-fast.json", start)
        self.assertIn('"minimax_music3.mem_saver": "false"', fast_config)
        self.assertIn('"minimax_music3.mem_saver": "true"', low_memory_config)

    def test_web_url_capture_contract_is_unchanged(self):
        start = (ROOT / "start.js").read_text(encoding="utf-8")
        self.assertIn('event: "/(http:\\\/\\\\/[0-9.:]+)/"', start)
        self.assertIn('url: "{{input.event[1]}}"', start)

    def test_app_installs_comfy_progress_transport(self):
        requirements = (ROOT / "app" / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^websockets(?:[<>=].*)?$")


if __name__ == "__main__":
    unittest.main()
