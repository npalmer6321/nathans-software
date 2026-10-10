package forgegpt.gui

import javax.swing.SwingUtilities
import javax.swing.UIManager

/**
 * GUI bootstrap, called from the CLI dispatcher (`forgegpt.jar gui`).
 *
 * Deliberately NOT a second `fun main()`: two entry points in one module
 * stop kotlinc from writing Main-Class into the JAR manifest, which
 * would break `java -jar forgegpt.jar train/generate/gradcheck`.
 * Sets a native-ish look and feel, then builds the window on the EDT.
 */
fun launchGui() {
    try {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())
    } catch (ignored: Exception) {
        // fall back to the default Swing look and feel
    }
    SwingUtilities.invokeLater { ForgeGptApp.launch() }
}
