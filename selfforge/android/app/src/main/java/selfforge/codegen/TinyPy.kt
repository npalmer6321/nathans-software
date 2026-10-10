package selfforge.codegen

/**
 * TinyPy: a miniature Python-like language the code model learns to write,
 * and a real interpreter that runs it. This is what "run the scripts" means
 * on-device — a small, safe sandbox for the model's own programs.
 *
 * Supported:
 *   x = 3 + 4 * (y - 1)     # assignment, ints, + - * // % and parens
 *   print(x)                 # print an expression
 *   # comments and blank lines are ignored
 *
 * Everything else is a syntax error. Division is floor division (//);
 * a single / is rejected to keep programs integer-clean.
 */
class TinyPy(private val maxOps: Int = 100_000) {

    sealed interface Result {
        data class Ok(val output: List<String>) : Result
        data class Err(val message: String) : Result
    }

    fun run(program: String): Result {
        val env = mutableMapOf<String, Long>()
        val out = mutableListOf<String>()
        var ops = 0
        fun bump() { ops++; if (ops > maxOps) throw ParseErr("too many operations") }
        program.lines().forEachIndexed { i, raw ->
            val line = raw.trim()
            if (line.isEmpty() || line.startsWith("#")) return@forEachIndexed
            val lineNo = i + 1
            try {
                when {
                    line.startsWith("print") -> {
                        val expr = line.removePrefix("print").trim()
                        if (!expr.startsWith("(") || !expr.endsWith(")"))
                            throw ParseErr("print needs parentheses")
                        out.add(eval(expr.drop(1).dropLast(1), env, lineNo, ::bump).toString())
                    }
                    line.contains("=") -> {
                        val parts = line.split("=", limit = 2)
                        val name = parts[0].trim()
                        if (!name.matches(Regex("[a-z]+"))) throw ParseErr("bad variable name '$name'")
                        env[name] = eval(parts[1], env, lineNo, ::bump)
                    }
                    else -> throw ParseErr("unknown statement")
                }
            } catch (e: ParseErr) {
                return Result.Err("line $lineNo: ${e.message}")
            }
        }
        return Result.Ok(out)
    }

    private class ParseErr(msg: String) : Exception(msg)

    // ---- expression parser (recursive descent) ----
    private class Lexer(val s: String) {
        var pos = 0
        fun peek(): Char? = if (pos < s.length) s[pos] else null
        fun next(): Char? = if (pos < s.length) s[pos++] else null
        fun skipWs() { while (peek()?.isWhitespace() == true) pos++ }
    }

    private fun eval(src: String, env: Map<String, Long>, lineNo: Int, bump: () -> Unit): Long {
        val lx = Lexer(src)
        fun parseExpr(): Long {
            var v = parseTerm(lx, env, lineNo, bump)
            while (true) {
                lx.skipWs()
                when (lx.peek()) {
                    '+' -> { lx.next(); bump(); v += parseTerm(lx, env, lineNo, bump) }
                    '-' -> { lx.next(); bump(); v -= parseTerm(lx, env, lineNo, bump) }
                    else -> return v
                }
            }
        }
        val v = parseExpr()
        lx.skipWs()
        if (lx.peek() != null) throw ParseErr("unexpected '${lx.peek()}' in expression")
        return v
    }

    private fun parseTerm(lx: Lexer, env: Map<String, Long>, lineNo: Int, bump: () -> Unit): Long {
        var v = parseFactor(lx, env, lineNo, bump)
        while (true) {
            lx.skipWs()
            when {
                lx.peek() == '*' -> { lx.next(); bump(); v *= parseFactor(lx, env, lineNo, bump) }
                lx.peek() == '%' -> { lx.next(); bump(); val d = parseFactor(lx, env, lineNo, bump); if (d == 0L) throw ParseErr("modulo by zero"); v %= d }
                lx.peek() == '/' -> {
                    lx.next()
                    if (lx.peek() != '/') throw ParseErr("use // for division")
                    lx.next(); bump()
                    val d = parseFactor(lx, env, lineNo, bump)
                    if (d == 0L) throw ParseErr("division by zero")
                    v = Math.floorDiv(v, d)
                }
                else -> return v
            }
        }
    }

    private fun parseFactor(lx: Lexer, env: Map<String, Long>, lineNo: Int, bump: () -> Unit): Long {
        lx.skipWs()
        val c = lx.peek() ?: throw ParseErr("unexpected end of expression")
        return when {
            c == '-' -> { lx.next(); -parseFactor(lx, env, lineNo, bump) }
            c == '(' -> {
                lx.next()
                // parse inner expr with a fresh lexer over the remaining source is awkward;
                // instead parse manually: find matching paren by scanning tokens.
                // Simpler: recursive parse on the substring.
                val start = lx.pos
                var depth = 1
                while (depth > 0) {
                    val ch = lx.next() ?: throw ParseErr("unclosed parenthesis")
                    if (ch == '(') depth++
                    if (ch == ')') depth--
                }
                val inner = lx.s.substring(start, lx.pos - 1)
                eval(inner, env, lineNo, bump)
            }
            c.isDigit() -> {
                val start = lx.pos
                while (lx.peek()?.isDigit() == true) lx.next()
                lx.s.substring(start, lx.pos).toLongOrNull() ?: throw ParseErr("bad number")
            }
            c.isLetter() -> {
                val start = lx.pos
                while (lx.peek()?.isLetter() == true) lx.next()
                val name = lx.s.substring(start, lx.pos)
                env[name] ?: throw ParseErr("unknown variable '$name'")
            }
            else -> throw ParseErr("unexpected '$c'")
        }
    }
}
