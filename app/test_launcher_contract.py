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

        install = (ROOT / "install.js").read_text(encoding="utf-8")
        self.assertIn(AUDIOCPP_SERVER_SHA256, install)
        self.assertNotIn("raw.githubusercontent.com/cocktailpeanut/minimax-music", install)

    def test_reset_keeps_bundled_server(self):
        reset = (ROOT / "reset.js").read_text(encoding="utf-8")
        self.assertNotIn("app/audiocpp_server", reset)


if __name__ == "__main__":
    unittest.main()
