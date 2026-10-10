package selfforge

/**
 * A "skill" = one small neural network plus the examples it has learned from.
 * Teaching the app a new ability means creating a new Skill and giving it
 * examples — the app then trains (and can self-train) that skill on-device.
 */
class Skill internal constructor(
    val name: String,
    val inputSize: Int,
    val labels: List<String>,
    val net: NeuralNet
) {
    val examples = mutableListOf<Example>()

    constructor(
        name: String,
        inputSize: Int,
        labels: List<String>,
        hiddenSize: Int = 24,
        seed: Long = 42L,
        learningRate: Double = 0.05
    ) : this(name, inputSize, labels, NeuralNet(intArrayOf(inputSize, hiddenSize, labels.size), learningRate, seed))

    /** Teach by example: "when you see this input, the answer is this label". */
    fun teach(input: DoubleArray, label: String) {
        val idx = labels.indexOf(label)
        require(idx >= 0) { "unknown label '$label' for skill '$name'" }
        require(input.size == inputSize) { "expected $inputSize inputs, got ${input.size}" }
        examples.add(Example(input, idx))
    }

    /** Plain supervised training on everything taught so far. */
    fun train(epochs: Int = 500) = net.trainBatch(examples, epochs)

    /**
     * Recursive self-training: learns from [unlabeled] inputs it labels itself,
     * judged against [validation]. Returns the per-round report.
     */
    fun selfTrain(
        unlabeled: List<DoubleArray>,
        validation: List<Example>,
        trainer: SelfTrainer = SelfTrainer()
    ): List<RoundReport> = trainer.selfTrain(net, examples.toList(), unlabeled, validation)

    /** Use the skill: get its best-guess label for an input. */
    fun guess(input: DoubleArray): String = labels[net.predict(input)]

    fun guessWithConfidence(input: DoubleArray): Pair<String, Double> {
        val p = net.predictProbs(input)
        var best = 0
        for (j in 1 until p.size) if (p[j] > p[best]) best = j
        return labels[best] to p[best]
    }

    fun accuracy(examples: List<Example>): Double = net.accuracy(examples)

    /** Save skill (metadata + weights + examples) to a portable string. */
    fun save(): String {
        val sb = StringBuilder()
        sb.append("SKILLv1\n")
        sb.append(name.replace("\n", " ")).append('\n')
        sb.append(inputSize).append('\n')
        sb.append(labels.joinToString("|")).append('\n')
        sb.append(examples.size).append('\n')
        for (e in examples) {
            sb.append(e.label).append(':').append(e.input.joinToString(",")).append('\n')
        }
        sb.append(net.exportWeights())
        return sb.toString()
    }

    companion object {
        fun load(data: String): Skill {
            val lines = data.trim().lines()
            require(lines[0] == "SKILLv1") { "not a skill file" }
            val name = lines[1]
            val inputSize = lines[2].toInt()
            val labels = lines[3].split("|")
            val nExamples = lines[4].toInt()
            var i = 5
            val examples = mutableListOf<Example>()
            repeat(nExamples) {
                val line = lines[i++]
                val label = line.substringBefore(':').toInt()
                val input = line.substringAfter(':').split(",").map { it.toDouble() }.toDoubleArray()
                examples.add(Example(input, label))
            }
            val net = NeuralNet.importWeights(lines.drop(i).joinToString("\n"))
            val skill = Skill(name, inputSize, labels, net)
            skill.examples.addAll(examples)
            return skill
        }
    }
}

/** Registry of everything the app has learned to do. New abilities = new skills. */
class SkillRegistry {
    private val skills = LinkedHashMap<String, Skill>()

    fun newSkill(
        name: String,
        inputSize: Int,
        labels: List<String>,
        hiddenSize: Int = 24,
        seed: Long = System.currentTimeMillis()
    ): Skill {
        require(!skills.containsKey(name)) { "skill '$name' already exists" }
        val s = Skill(name, inputSize, labels, hiddenSize, seed)
        skills[name] = s
        return s
    }

    fun get(name: String): Skill? = skills[name]
    fun names(): List<String> = skills.keys.toList()
    fun remove(name: String): Boolean = skills.remove(name) != null
    fun count(): Int = skills.size
}
