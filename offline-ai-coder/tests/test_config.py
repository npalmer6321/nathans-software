import json
import tempfile
import unittest
from pathlib import Path

from app import config


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "settings.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_are_valid(self):
        self.assertEqual(config.Settings().validate(), [])

    def test_active_model_roles(self):
        s = config.Settings(main_model="m1", fast_model="f1")
        self.assertEqual(s.active_model, "m1")
        s.default_model_role = "fast"
        self.assertEqual(s.active_model, "f1")

    def test_save_load_round_trip(self):
        s = config.Settings(main_model="custom:1", theme="light", font_size=13)
        config.save_settings(s, self.path)
        loaded = config.load_settings(self.path)
        self.assertFalse(loaded.created_new)
        self.assertEqual(loaded.settings.main_model, "custom:1")
        self.assertEqual(loaded.settings.theme, "light")
        self.assertEqual(loaded.settings.font_size, 13)

    def test_missing_file_gives_defaults(self):
        loaded = config.load_settings(self.path)
        self.assertTrue(loaded.created_new)
        self.assertIsNone(loaded.warning)

    def test_corrupt_file_gives_defaults_and_backup(self):
        self.path.write_text("{not json", encoding="utf-8")
        loaded = config.load_settings(self.path)
        self.assertIsNotNone(loaded.warning)
        self.assertEqual(loaded.settings.main_model, config.DEFAULT_MAIN_MODEL)
        self.assertTrue((Path(self.tmp.name) / "settings.json.bak").exists())

    def test_unknown_keys_survive_round_trip(self):
        data = {"main_model": "x", "future_option": {"a": 1}}
        s = config.settings_from_dict(data)
        self.assertEqual(s.extra, {"future_option": {"a": 1}})
        self.assertEqual(config.settings_to_dict(s)["future_option"], {"a": 1})

    def test_save_rejects_invalid_settings(self):
        s = config.Settings(ollama_base_url="ftp://nope")
        with self.assertRaises(config.ConfigError):
            config.save_settings(s, self.path)

    def test_recent_projects_dedupes_and_caps(self):
        s = config.Settings(max_recent_projects=3)
        for name in ("a", "b", "c"):
            s.remember_project(name)
        s.remember_project("a")
        self.assertEqual(s.recent_projects, ["a", "c", "b"])
        self.assertEqual(s.last_project_dir, "a")

    def test_validation_catches_bad_values(self):
        s = config.Settings(temperature=9.0, num_ctx=10, theme="plaid")
        problems = s.validate()
        self.assertTrue(any("Temperature" in p for p in problems))
        self.assertTrue(any("Context" in p for p in problems))
        self.assertTrue(any("Theme" in p for p in problems))

    def test_file_contents_are_json(self):
        config.save_settings(config.Settings(), self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("ollama_base_url", data)


if __name__ == "__main__":
    unittest.main()
