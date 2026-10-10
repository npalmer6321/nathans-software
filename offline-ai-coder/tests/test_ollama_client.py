import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from app import ollama_client as oc
from app.errors import OllamaHTTPError, OllamaProtocolError


class ParseTests(unittest.TestCase):
    def test_human_bytes(self):
        self.assertEqual(oc.human_bytes(512), "512 B")
        self.assertTrue(oc.human_bytes(2048).endswith("KB"))
        self.assertTrue(oc.human_bytes(5 * 1024 ** 3).endswith("GB"))

    def test_parse_tags(self):
        payload = {"models": [
            {"name": "b:7b", "size": 4000, "details": {"family": "qwen"}},
            {"name": "a:1b", "size": 1000},
            {"no_name": True},
        ]}
        models = oc.parse_tags_payload(payload)
        self.assertEqual([m.name for m in models], ["a:1b", "b:7b"])
        self.assertEqual(models[1].family, "qwen")
        self.assertEqual(models[1].base_name, "b")
        self.assertEqual(models[1].tag, "7b")

    def test_parse_tags_rejects_junk(self):
        with self.assertRaises(OllamaProtocolError):
            oc.parse_tags_payload([1, 2])

    def test_parse_pull_line_fraction(self):
        msg, frac = oc.parse_pull_line('{"status":"pulling","total":100,"completed":25}')
        self.assertAlmostEqual(frac, 0.25)
        self.assertIn("25 B", msg)

    def test_parse_pull_line_plain(self):
        msg, frac = oc.parse_pull_line('{"status":"verifying"}')
        self.assertEqual(msg, "verifying")
        self.assertIsNone(frac)

    def test_parse_chat_payload_message(self):
        result = oc.parse_chat_payload({"model": "m", "message": {"content": "hi"},
                                        "total_duration": 2_000_000_000, "eval_count": 50})
        self.assertEqual(result.content, "hi")
        self.assertAlmostEqual(result.tokens_per_second, 25.0)

    def test_parse_chat_payload_generate_style(self):
        result = oc.parse_chat_payload({"response": "done"}, model="x")
        self.assertEqual(result.content, "done")

    def test_iter_json_lines_skips_blanks_and_bad(self):
        lines = iter([b'{"a": 1}\n', b"\n", b'{"b": 2}'])
        objs = list(oc.iter_json_lines(lines))
        self.assertEqual(objs, [{"a": 1}, {"b": 2}])
        with self.assertRaises(OllamaProtocolError):
            list(oc.iter_json_lines([b"not json"]))


class StubHandler(BaseHTTPRequestHandler):
    """Tiny fake Ollama for offline client tests."""

    def log_message(self, *args):  # keep the test output clean
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._json({"version": "0.9.9"})
        elif self.path == "/api/tags":
            self._json({"models": [{"name": "qwen:7b", "size": 1234}]})
        else:
            self._json({"version": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/api/chat":
            self._json({"model": "qwen:7b", "message": {"role": "assistant",
                                                        "content": "hello back"}})
        elif self.path == "/api/pull":
            lines = (b'{"status":"pulling","total":10,"completed":5}\n'
                     b'{"status":"success"}\n')
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(lines)))
            self.end_headers()
            self.wfile.write(lines)
        elif self.path == "/api/show":
            self._json({"details": {"family": "qwen2"}})
        else:
            self._json({"error": "nope"}, code=404)

    def do_DELETE(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._json({}, code=200)


class ClientAgainstStubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), StubHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = oc.OllamaClient(f"http://127.0.0.1:{cls.port}", timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_status(self):
        status = self.client.status()
        self.assertTrue(status.reachable)
        self.assertEqual(status.version, "0.9.9")
        self.assertEqual(status.model_count, 1)

    def test_list_models(self):
        models = self.client.list_models()
        self.assertEqual(models[0].name, "qwen:7b")

    def test_chat(self):
        result = self.client.chat("qwen:7b", [oc.ChatMessage("user", "hi")])
        self.assertEqual(result.content, "hello back")

    def test_pull_progress(self):
        seen = []
        last = self.client.pull_model("qwen:7b", progress=lambda m, f: seen.append((m, f)))
        self.assertEqual(last, "success")
        self.assertTrue(any(frac == 0.5 for _, frac in seen))

    def test_show_and_delete(self):
        info = self.client.show_model("qwen:7b")
        self.assertEqual(info["details"]["family"], "qwen2")
        self.client.delete_model("qwen:7b")  # no error means success

    def test_unreachable_client(self):
        dead = oc.OllamaClient("http://127.0.0.1:1", timeout=1)
        self.assertFalse(dead.ping())
        self.assertFalse(dead.status().reachable)
        with self.assertRaises(Exception):
            dead.list_models()


if __name__ == "__main__":
    unittest.main()
