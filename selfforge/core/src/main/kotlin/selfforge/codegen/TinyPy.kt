package selfforge.codegen

/**
 * TinyPy: a miniature Python-like language the code model learns to write,
 * and a real interpreter that runs it. This is what "run the scripts" means
 * on-device — a small, safe sandbox for the model's own programs.
 *
 * Supported:
 *   x = 3 + 4 * (y - 1)     # assignment; ints; + - * // %
 *   print(x)                 # print an expression
 *   if x > 2:                # comparisons: > < >= <= == != (1 = true, 0 = false)
 *       y = 1                # indented blocks (spaces)
 *   if x <= 2:              # (no else — two ifs cover it)
 *       y = 2
 *   while x > 0:             # loops (capped so runaway loops fail safely)
 *       x = x - 1
 *
 * Deliberately no `else`: the tiny model learns local patterns well but
 * couldn't reliably learn the long-range "else must follow an if" structure,
 * so two complementary `if`s are used instead (and it learns those fine).
 *
 * Everything else is an error. Division is floor division (//).
 */
class TinyPy(private val maxOps: Int = 200_000, private val maxLoopIter: Int = 20_000) {

    sealed interface Result {
        data class Ok(val output: List<String>) : Result
        data class Err(val message: String) : Result
    }

    fun run(program: String): Result {
        return try {
            val lines = program.lines().mapIndexedNotNull { i, raw ->
                val code = raw.substringBefore("#")
                if (code.isBlank()) null
                else {
                    if (code.first() != ' ' && code.first().isWhitespace())
                        throw ParseErr("line ${i + 1}: use spaces for indentation")
                    Line(code.takeWhile { it == ' ' }.length, code.trim(), i + 1)
                }
            }
            val (stmts, pos) = parseBlock(lines, 0, -1)
            if (pos != lines.size) throw ParseErr("line ${lines[pos].lineNo}: unexpected indentation")
            val env = mutableMapOf<String, Long>()
            val out = mutableListOf<String>()
            var ops = 0
            fun bump() { if (++ops > maxOps) throw ParseErr("too many operations") }
            exec(stmts, env, out, ::bump)
            Result.Ok(out)
        } catch (e: ParseErr) {
            Result.Err(e.message ?: "parse error")
        }
    }

    private class ParseErr(msg: String) : Exception(msg)
    private data class Line(val indent: Int, val text: String, val lineNo: Int)

    private sealed interface Stmt {
        data class Assign(val name: String, val expr: Expr) : Stmt
        data class Print(val expr: Expr) : Stmt
        data class If(val cond: Expr, val thenB: List<Stmt>, val elseB: List<Stmt>) : Stmt
        data class While(val cond: Expr, val body: List<Stmt>) : Stmt
    }

    private sealed interface Expr {
        data class Num(val v: Long) : Expr
        data class Var(val name: String) : Expr
        data class Neg(val e: Expr) : Expr
        data class Bin(val op: String, val l: Expr, val r: Expr) : Expr
        data class Cmp(val op: String, val l: Expr, val r: Expr) : Expr
    }

    // ---------- block parser ----------
    private fun parseBlock(lines: List<Line>, from: Int, parentIndent: Int): Pair<List<Stmt>, Int> {
        if (from >= lines.size || lines[from].indent <= parentIndent) return emptyList<Stmt>() to from
        val level = lines[from].indent
        val stmts = mutableListOf<Stmt>()
        var p = from
        while (p < lines.size && lines[p].indent == level) {
            val (stmt, next) = parseStmt(lines, p, level)
            stmts.add(stmt)
            p = next
        }
        return stmts to p
    }

    private fun parseStmt(lines: List<Line>, p: Int, level: Int): Pair<Stmt, Int> {
        val ln = lines[p]
        val text = ln.text
        if (text.endsWith(":")) {
            val header = text.dropLast(1).trim()
            when {
                header.startsWith("if ") -> {
                    val cond = parseExprStr(header.removePrefix("if").trim(), ln.lineNo)
                    val (thenB, p2) = parseBlock(lines, p + 1, level)
                    if (thenB.isEmpty()) throw ParseErr("line ${ln.lineNo}: expected an indented block after if")
                    // No `else`: deliberately omitted. The tiny model learns local
                    // patterns well but can't reliably track the long-range
                    // "else must follow an if" structure, so `else:` is an error.
                    // Two complementary `if`s cover the same ground.
                    return Stmt.If(cond, thenB, emptyList()) to p2
                }
                header.startsWith("while ") -> {
                    val cond = parseExprStr(header.removePrefix("while").trim(), ln.lineNo)
                    val (body, p2) = parseBlock(lines, p + 1, level)
                    if (body.isEmpty()) throw ParseErr("line ${ln.lineNo}: expected an indented block after while")
                    return Stmt.While(cond, body) to p2
                }
                else -> throw ParseErr("line ${ln.lineNo}: unknown block header")
            }
        }
        val printRest = if (text.startsWith("print")) text.drop(5).trim() else ""
        return when {
            text.startsWith("print") && printRest.startsWith("(") && printRest.endsWith(")") ->
                Stmt.Print(parseExprStr(printRest.drop(1).dropLast(1), ln.lineNo)) to p + 1
            text.contains("=") -> {
                val idx = text.indexOf("=")
                val name = text.substring(0, idx).trim()
                if (!name.matches(Regex("[a-z]+"))) throw ParseErr("line ${ln.lineNo}: bad variable name")
                Stmt.Assign(name, parseExprStr(text.substring(idx + 1), ln.lineNo)) to p + 1
            }
            else -> throw ParseErr("line ${ln.lineNo}: unknown statement")
        }
    }

    // ---------- expression parser ----------
    private class Lex(val s: String) {
        var pos = 0
        fun peek(): Char? = s.getOrNull(pos)
        fun peek2(): Char? = s.getOrNull(pos + 1)
        fun next(): Char? = s.getOrNull(pos++)
        fun skipWs() { while (peek()?.let { it == ' ' || it == '\t' } == true) pos++ }
    }

    private fun parseExprStr(src: String, lineNo: Int): Expr {
        val lx = Lex(src)
        val e = parseCmp(lx, lineNo)
        lx.skipWs()
        if (lx.peek() != null) throw ParseErr("line $lineNo: unexpected '${lx.peek()}'")
        return e
    }

    private fun parseCmp(lx: Lex, lineNo: Int): Expr {
        var v = parseAdd(lx, lineNo)
        while (true) {
            lx.skipWs()
            val op = when {
                lx.peek() == '=' && lx.peek2() == '=' -> "=="
                lx.peek() == '!' && lx.peek2() == '=' -> "!="
                lx.peek() == '>' && lx.peek2() == '=' -> ">="
                lx.peek() == '<' && lx.peek2() == '=' -> "<="
                lx.peek() == '>' -> ">"
                lx.peek() == '<' -> "<"
                else -> return v
            }
            repeat(op.length) { lx.next() }
            v = Expr.Cmp(op, v, parseAdd(lx, lineNo))
        }
    }

    private fun parseAdd(lx: Lex, lineNo: Int): Expr {
        var v = parseMul(lx, lineNo)
        while (true) {
            lx.skipWs()
            when (lx.peek()) {
                '+' -> { lx.next(); v = Expr.Bin("+", v, parseMul(lx, lineNo)) }
                '-' -> { lx.next(); v = Expr.Bin("-", v, parseMul(lx, lineNo)) }
                else -> return v
            }
        }
    }

    private fun parseMul(lx: Lex, lineNo: Int): Expr {
        var v = parsePrimary(lx, lineNo)
        while (true) {
            lx.skipWs()
            when {
                lx.peek() == '*' -> { lx.next(); v = Expr.Bin("*", v, parsePrimary(lx, lineNo)) }
                lx.peek() == '%' -> { lx.next(); v = Expr.Bin("%", v, parsePrimary(lx, lineNo)) }
                lx.peek() == '/' -> {
                    lx.next()
                    if (lx.peek() != '/') throw ParseErr("line $lineNo: use // for division")
                    lx.next()
                    v = Expr.Bin("//", v, parsePrimary(lx, lineNo))
                }
                else -> return v
            }
        }
    }

    private fun parsePrimary(lx: Lex, lineNo: Int): Expr {
        lx.skipWs()
        val c = lx.peek() ?: throw ParseErr("line $lineNo: unexpected end of expression")
        return when {
            c == '-' -> { lx.next(); Expr.Neg(parsePrimary(lx, lineNo)) }
            c == '(' -> {
                lx.next()
                val e = parseCmp(lx, lineNo)
                lx.skipWs()
                if (lx.next() != ')') throw ParseErr("line $lineNo: expected ')'")
                e
            }
            c.isDigit() -> {
                val s = lx.pos
                while (lx.peek()?.isDigit() == true) lx.next()
                Expr.Num(lx.s.substring(s, lx.pos).toLongOrNull()
                    ?: throw ParseErr("line $lineNo: bad number"))
            }
            c.isLetter() -> {
                val s = lx.pos
                while (lx.peek()?.isLetter() == true) lx.next()
                Expr.Var(lx.s.substring(s, lx.pos))
            }
            else -> throw ParseErr("line $lineNo: unexpected '$c'")
        }
    }

    // ---------- interpreter ----------
    private fun exec(stmts: List<Stmt>, env: MutableMap<String, Long>, out: MutableList<String>, bump: () -> Unit) {
        for (s in stmts) when (s) {
            is Stmt.Assign -> env[s.name] = eval(s.expr, env, bump)
            is Stmt.Print -> out.add(eval(s.expr, env, bump).toString())
            is Stmt.If -> if (eval(s.cond, env, bump) != 0L) exec(s.thenB, env, out, bump)
                           else exec(s.elseB, env, out, bump)
            is Stmt.While -> {
                var iter = 0
                while (eval(s.cond, env, bump) != 0L) {
                    if (++iter > maxLoopIter) throw ParseErr("loop ran too long")
                    exec(s.body, env, out, bump)
                }
            }
        }
    }

    private fun eval(e: Expr, env: Map<String, Long>, bump: () -> Unit): Long {
        bump()
        return when (e) {
            is Expr.Num -> e.v
            is Expr.Var -> env[e.name] ?: throw ParseErr("unknown variable '${e.name}'")
            is Expr.Neg -> -eval(e.e, env, bump)
            is Expr.Bin -> {
                val l = eval(e.l, env, bump)
                val r = eval(e.r, env, bump)
                when (e.op) {
                    "+" -> l + r
                    "-" -> l - r
                    "*" -> l * r
                    "%" -> if (r == 0L) throw ParseErr("modulo by zero") else l % r
                    "//" -> if (r == 0L) throw ParseErr("division by zero") else Math.floorDiv(l, r)
                    else -> throw ParseErr("bad operator")
                }
            }
            is Expr.Cmp -> {
                val l = eval(e.l, env, bump)
                val r = eval(e.r, env, bump)
                val t = when (e.op) {
                    ">" -> l > r; "<" -> l < r; ">=" -> l >= r
                    "<=" -> l <= r; "==" -> l == r; "!=" -> l != r
                    else -> throw ParseErr("bad comparison")
                }
                if (t) 1 else 0
            }
        }
    }
}
