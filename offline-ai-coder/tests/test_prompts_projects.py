import tempfile
import unittest
from pathlib import Path

from app import projects as projects_mod
from app.prompts import PromptError, PromptLibrary, PromptTemplate


class PromptTests(unittest.TestCase):
    def test_builtins_present(self):
        lib = PromptLibrary()
        self.assertGreaterEqual(len(lib.all()), 8)
        self.assertTrue(all(p.builtin for p in lib.all()))

    def test_variables_and_render(self):
        t = PromptTemplate(id="t", title="t", category="c",
                           body="Hi {name}, fix {thing} and {thing}.")
        self.assertEqual(t.variables, ["name", "thing"])
        self.assertEqual(t.render({"name": "Bo"}), "Hi Bo, fix {thing} and {thing}.")
        self.assertEqual(t.render_strict({"name": "Bo", "thing": "it"}),
                         "Hi Bo, fix it and it.")

    def test_render_strict_missing(self):
        t = PromptTemplate(id="t", title="t", category="c", body="Need {x}")
        with self.assertRaises(PromptError):
            t.render_strict({})

    def test_crud_and_persistence(self):
        lib = PromptLibrary()
        mine = lib.add("Mine", "Cat", "Body {v}")
        self.assertIn(mine, lib.all())
        lib.update(mine.id, "Mine v2", "Cat", "Body2")
        self.assertEqual(lib.get(mine.id).title, "Mine v2")
        lib.delete(mine.id)
        self.assertEqual(lib.search("Mine v2"), [])

    def test_builtin_delete_hides_and_restores(self):
        lib = PromptLibrary()
        first = lib.all()[0]
        lib.delete(first.id)
        self.assertNotIn(first, lib.all())
        self.assertIn(first, lib.all(include_hidden=True))
        lib.restore_builtins()
        self.assertIn(first, lib.all())

    def test_search_and_categories(self):
        lib = PromptLibrary()
        hits = lib.search("error")
        self.assertTrue(any("error" in (p.title + p.body).lower() for p in hits))
        self.assertIn("Debugging", lib.categories())

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            lib = PromptLibrary()
            lib.add("Saved", "C", "text")
            lib.save(path)
            again = PromptLibrary.load(path)
            self.assertTrue(any(p.title == "Saved" for p in again.all()))

    def test_builtin_cannot_be_edited(self):
        lib = PromptLibrary()
        with self.assertRaises(PromptError):
            lib.update(lib.all()[0].id, "x", "y", "z")


class ProjectTests(unittest.TestCase):
    def _make_tree(self, root: Path) -> None:
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
        (root / "src" / "util.py").write_text("X = 1\n", encoding="utf-8")
        (root / "test_main.py").write_text("import unittest\n", encoding="utf-8")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
        (root / "README.md").write_text("# Demo\n", encoding="utf-8")

    def test_scan_skips_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            project = projects_mod.scan_project(root)
            names = [rel for rel, _ in project.tree.flatten()]
            self.assertIn("src/main.py", names)
            self.assertFalse(any("pycache" in n for n in names))

    def test_summary_and_tests_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            project = projects_mod.scan_project(root)
            self.assertEqual(project.file_count, 4)
            self.assertIn(".py", project.language_summary())
            tests = projects_mod.find_test_files(project)
            self.assertTrue(any(p.name == "test_main.py" for p in tests))

    def test_read_preview_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp) / "a.txt"
            text.write_text("hello", encoding="utf-8")
            preview = projects_mod.read_preview(text)
            self.assertEqual(preview.text, "hello")
            self.assertFalse(preview.binary)
            blob = Path(tmp) / "b.bin"
            blob.write_bytes(b"\x00\x01\x02")
            self.assertTrue(projects_mod.read_preview(blob).binary)

    def test_read_preview_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.txt"
            big.write_text("x" * 5000, encoding="utf-8")
            preview = projects_mod.read_preview(big, max_bytes=100)
            self.assertTrue(preview.truncated)
            self.assertEqual(len(preview.text), 100)

    def test_scan_missing_folder(self):
        with self.assertRaises(projects_mod.ProjectError):
            projects_mod.scan_project("/definitely/not/here-xyz")

    def test_context_includes_small_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            project = projects_mod.scan_project(root)
            context = projects_mod.project_context(project)
            self.assertIn("Project:", context)
            self.assertIn("main.py", context)


if __name__ == "__main__":
    unittest.main()
