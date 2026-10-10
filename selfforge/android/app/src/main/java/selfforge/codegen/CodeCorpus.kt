package selfforge.codegen

/**
 * The textbook the code model learns from: short, correct programs in the
 * TinyPy subset. Variable names stay consistent (x, y, z) so the model can
 * learn the pattern of defining a name before using it. The model learns
 * their shape purely from next-character prediction — nobody ever tells it
 * the grammar rules.
 */
object CodeCorpus {
    val programs: List<String> = listOf(
        "x = 5\nprint(x)\n",
        "x = 3\ny = 7\nprint(x + y)\n",
        "x = 10\ny = 4\nprint(x - y)\n",
        "x = 6\ny = 7\nprint(x * y)\n",
        "x = 20\nprint(x // 3)\n",
        "x = 17\nprint(x % 5)\n",
        "x = 2\ny = 3\nz = x + y * 2\nprint(z)\n",
        "x = 9\ny = 2\nz = (x + y) * 3\nprint(z)\n",
        "x = 100\ny = x // 7\nprint(y)\n",
        "x = 1\ny = 2\nz = x + y + 3\nprint(z)\n",
        "x = 8\nprint(x * x)\n",
        "x = 5\ny = x + x\nprint(y)\n",
        "x = 12\ny = 5\nprint(x % y)\n",
        "x = 7\ny = x * x\nprint(y)\n",
        "x = 4\ny = x + 1\nz = y + 1\nprint(z)\n",
        "x = 15\nprint(x // 4)\n",
        "x = 3\ny = 4\nprint(x * x + y * y)\n",
        "x = 11\ny = x - 6\nprint(y * 2)\n",
        "x = 9\ny = 3\nprint((x + y) // 2)\n",
        "x = 6\nprint(x * (x + 1))\n",
        "x = 2\ny = 8\nprint(y // x)\n",
        "x = 14\ny = x % 4\nprint(y)\n",
        "x = 5\ny = 6\nz = x * y - 10\nprint(z)\n",
        "x = 1\ny = 2\nz = 3\nprint(x + y + z)\n",
        "x = 25\nprint(x // 5)\n",
        "x = 7\ny = x * 3\nprint(y - x)\n",
        "x = 4\ny = 9\nprint(x * y)\n",
        "x = 30\nprint(x % 7)\n",
        "x = 3\ny = 4\nz = 5\nprint(x + y + z)\n",
        "x = 16\ny = x // 2\nz = y // 2\nprint(z)\n",
        "x = 8\ny = x + 12\nprint(y)\n",
        "x = 5\ny = 2\nprint(x - y)\n",
        "x = 7\ny = 7\nprint(x * y)\n",
        "x = 42\nprint(x % 10)\n",
        "x = 2\ny = 10\nprint(y // x)\n",
        "x = 3\ny = x * x * x\nprint(y)\n",
        "x = 18\ny = 4\nprint(x // y)\n",
        "x = 1\ny = 9\nprint((x + y) * (x + y))\n",
        "x = 64\nprint(x // 8)\n",
        "x = 5\ny = x + 15\nprint(y % 6)\n",
        "x = 11\ny = 2\nprint(x * y + 1)\n",
        "x = 9\nprint(x * x - x)\n",
        "x = 4\ny = 6\nz = x * y\nprint(z // 3)\n",
        "x = 13\ny = 5\nprint(x % y)\n",
        "x = 10\ny = x - 3\nprint(y * y)\n",
        "x = 3\ny = 12\nprint(y // x + 1)\n",
        "x = 7\ny = 8\nz = 9\nprint(x * y - z)\n",
        "x = 20\ny = 6\nprint(x // y)\n",
        "x = 2\ny = 2\nprint(x + y * 10)\n",
        "x = 14\ny = 3\nprint(x % y)\n",
    )

    /** Every character the model will ever read or write. */
    val vocab: List<Char> = (programs.joinToString("").toSet()).sorted()
}
