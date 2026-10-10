import unittest

from stringtools import reverse_words


class TestReverseWords(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(reverse_words("the quick brown fox"), "fox brown quick the")

    def test_single_word(self):
        self.assertEqual(reverse_words("hello"), "hello")

    def test_extra_spaces(self):
        self.assertEqual(reverse_words("  one   two  "), "two one")


if __name__ == "__main__":
    unittest.main()
