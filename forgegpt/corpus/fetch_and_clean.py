#!/usr/bin/env python3
"""Fetch public-domain books from Project Gutenberg and clean them into corpus.txt.

Books (all US public domain):
  11    Alice's Adventures in Wonderland (Lewis Carroll)
  1342  Pride and Prejudice (Jane Austen)
  1661  The Adventures of Sherlock Holmes (Arthur Conan Doyle)
  84    Frankenstein (Mary Shelley)
  98    A Tale of Two Cities (Charles Dickens)
  2701  Moby Dick (Herman Melville)

Usage: python3 fetch_and_clean.py   (run inside the corpus/ directory)
"""
import re, urllib.request

BOOKS = [11, 1342, 1661, 84, 98, 2701]

def fetch(book_id):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print("fetching", url)
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")

def clean(t):
    m1 = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S)
    m2 = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S)
    if m1 and m2:
        t = t[m1.end():m2.start()]
    t = (t.replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2014", " -- ").replace("\u2013", " - ")
          .replace("\u2026", "..."))
    t = "".join(c if ord(c) < 128 else " " for c in t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

parts = [clean(fetch(b)) for b in BOOKS]
full = "\n\n".join(parts)
with open("corpus.txt", "w") as f:
    f.write(full)
print(f"wrote corpus.txt: {len(full)} chars, ~{len(full.split())} words")
