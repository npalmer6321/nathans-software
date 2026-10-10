package forgegpt

import java.io.File

/**
 * Word-level tokenizer. Vocabulary is built from the training corpus:
 * the most frequent tokens, plus four reserved specials.
 *
 * Token pattern: lowercase words (letters/digits, keeping internal
 * apostrophes like "don't") or single punctuation characters.
 */
class Tokenizer(val idToToken: List<String>) {
    companion object {
        const val PAD = 0
        const val UNK = 1
        const val BOS = 2
        const val EOS = 3
        val SPECIALS = listOf("<pad>", "<unk>", "<bos>", "<eos>")

        private val WORD = Regex("[a-z0-9]+(?:'[a-z]+)?")
        private val TOKEN = Regex("[a-z0-9]+(?:'[a-z]+)?|\\S")

        private val NO_SPACE_BEFORE = setOf(
            ".", ",", "!", "?", ":", ";", ")", "]", "}", "'", "\"", "%", "--", "''", "``"
        )
        private val NO_SPACE_AFTER = setOf("(", "[", "{", "\"", "'", "``", "\$")

        fun tokenizeText(text: String): List<String> =
            TOKEN.findAll(text.lowercase()).map { it.value }.toList()

        /** Build a vocab from raw corpus text: most frequent tokens win. */
        fun buildFromCorpus(text: String, maxVocab: Int, minCount: Int = 3): Tokenizer {
            val counts = HashMap<String, Int>()
            for (tok in tokenizeText(text)) counts[tok] = (counts[tok] ?: 0) + 1
            val kept = counts.entries
                .filter { it.value >= minCount }
                .sortedByDescending { it.value }
                .take((maxVocab - SPECIALS.size).coerceAtLeast(1))
                .map { it.key }
            return Tokenizer(SPECIALS + kept)
        }

        fun load(file: File): Tokenizer =
            Tokenizer(file.readLines().map { it.trimEnd('\n', '\r') }.filter { it.isNotEmpty() })
    }

    val tokenToId: Map<String, Int> = idToToken.withIndex().associate { it.value to it.index }
    val vocabSize: Int get() = idToToken.size

    fun encode(text: String): IntArray {
        val toks = tokenizeText(text)
        return IntArray(toks.size) { tokenToId[toks[it]] ?: UNK }
    }

    /** Encode with BOS at the front and EOS at the end. */
    fun encodeDocument(text: String): IntArray {
        val inner = encode(text)
        return IntArray(inner.size + 2) { i ->
            when (i) {
                0 -> BOS
                inner.size + 1 -> EOS
                else -> inner[i - 1]
            }
        }
    }

    fun decode(ids: IntArray): String = decode(ids.toList())

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        var prev = ""
        for (id in ids) {
            if (id < 0 || id >= idToToken.size) continue
            val tok = idToToken[id]
            if (tok in SPECIALS) continue // don't render specials
            if (sb.isEmpty()) {
                sb.append(tok)
            } else if (tok in NO_SPACE_BEFORE || prev in NO_SPACE_AFTER) {
                sb.append(tok)
            } else {
                sb.append(' ').append(tok)
            }
            prev = tok
        }
        return sb.toString()
    }

    fun save(file: File) {
        file.writeText(idToToken.joinToString("\n"))
    }
}
