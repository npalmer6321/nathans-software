import tempfile
import unittest
from pathlib import Path

from app import chatstore, logstore, system_info


class LogstoreTests(unittest.TestCase):
    def test_parse_line(self):
        line = logstore.parse_line("2026-10-10 01:00:00 [ERROR] it broke")
        self.assertEqual(line.level, "ERROR")
        self.assertTrue(line.is_error)
        self.assertEqual(line.message, "it broke")

    def test_parse_unformatted(self):
        line = logstore.parse_line("just some text")
        self.assertEqual(line.message, "just some text")
        self.assertFalse(line.is_error)

    def test_read_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.log"
            path.write_text(
                "2026-10-10 01:00:00 [INFO] started\n"
                "2026-10-10 01:01:00 [ERROR] server down\n", encoding="utf-8")
            lines = logstore.read_log(path)
            self.assertEqual(len(lines), 2)
            errors = logstore.search_log(lines, "", errors_only=True)
            self.assertEqual(len(errors), 1)
            found = logstore.search_log(lines, "server")
            self.assertEqual(len(found), 1)
            self.assertEqual(len(logstore.recent_errors(path)), 1)

    def test_missing_log(self):
        self.assertEqual(logstore.read_log(Path("/nope/nothing.log")), [])


class ChatstoreTests(unittest.TestCase):
    def test_session_titles_and_window(self):
        s = chatstore.new_session("m1")
        s.add("user", "how do I sort a list in Python?")
        self.assertIn("sort", s.title)
        self.assertEqual(s.turn_count, 1)
        payload = s.payload()
        self.assertEqual(payload[0]["role"], "system")
        self.assertEqual(payload[-1]["role"], "user")

    def test_clear_keeps_system(self):
        s = chatstore.new_session("m1")
        s.add("user", "hi")
        s.add("assistant", "hello")
        s.clear()
        self.assertEqual(len(s.messages), 1)
        self.assertEqual(s.messages[0].role, "system")

    def test_transcript_labels(self):
        s = chatstore.new_session("m1")
        s.add("user", "ping")
        s.add("assistant", "pong")
        text = s.transcript()
        self.assertIn("**You:**", text)
        self.assertIn("**AI:**", text)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chats.json"
            s = chatstore.new_session("m1")
            s.add("user", "question")
            s.add("assistant", "answer")
            chatstore.save_sessions([s], path)
            loaded = chatstore.load_sessions(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].messages[-1].content, "answer")

    def test_load_tolerates_junk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chats.json"
            path.write_text("{oops", encoding="utf-8")
            self.assertEqual(chatstore.load_sessions(path), [])

    def test_session_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chats.json"
            many = []
            for i in range(chatstore.MAX_STORED_SESSIONS + 5):
                s = chatstore.new_session("m")
                s.title = f"s{i}"
                s.updated = float(i)
                many.append(s)
            chatstore.save_sessions(many, path)
            loaded = chatstore.load_sessions(path)
            self.assertEqual(len(loaded), chatstore.MAX_STORED_SESSIONS)


class SystemInfoTests(unittest.TestCase):
    def test_snapshot_shape(self):
        snap = system_info.snapshot()
        self.assertTrue(snap.os_name)
        self.assertTrue(snap.python_version)
        self.assertIsInstance(snap.disks, list)

    def test_ram_meets_unknown(self):
        snap = system_info.SystemSnapshot("x", "1", "x86", "3.12", 4, None, [])
        self.assertIsNone(snap.ram_meets(16))
        snap2 = system_info.SystemSnapshot("x", "1", "x86", "3.12", 4, 32 * 1024 ** 3, [])
        self.assertTrue(snap2.ram_meets(16))
        self.assertEqual(snap2.total_ram_human, "32.0 GB")

    def test_format_snapshot(self):
        snap = system_info.SystemSnapshot(
            "TestOS", "1.0", "x86_64", "3.12.0", 8, 16 * 1024 ** 3,
            [system_info.DiskInfo("/home", 100, 40)])
        text = system_info.format_snapshot(snap)
        self.assertIn("TestOS", text)
        self.assertIn("free of", text)

    def test_ram_advice(self):
        snap = system_info.SystemSnapshot("x", "1", "x86", "3.12", 4, 8 * 1024 ** 3, [])
        advice = system_info.ram_advice("qwen2.5-coder:14b", snap)
        self.assertIn("Below", advice)
        self.assertIn("No specific", system_info.ram_advice("unknown:1b", snap))


if __name__ == "__main__":
    unittest.main()
