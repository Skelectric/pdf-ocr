import { tool } from "@opencode-ai/plugin"
import { resolve } from "path"
import { execFile } from "child_process"
import { promisify } from "util"

const execFileAsync = promisify(execFile)
const TOOL_DIR =
  process.env.PDF_OCR_TOOL_DIR || `${process.env.HOME}/.config/opencode/tool`

export default tool({
  description:
    "Extract text from PDF files through the configured OCR route. Processes entire PDFs or specific pages and returns markdown. Use this when you need to transcribe PDF documents for analysis or processing.",
  args: {
    pdf_path: tool.schema.string().describe("Path to PDF file (absolute or relative to current directory) or a URL to download the PDF from"),
    page: tool.schema.string().optional().describe("Page(s) to OCR: single page ('5'), range ('1-5'), multiple ('1,3,5'), or mixed ('1-3,5,7-9'). If omitted, all pages are processed.")
  },
  async execute(args) {
    // Handle both local paths and URLs
    const pdfPath = args.pdf_path.startsWith('http://') || args.pdf_path.startsWith('https://')
      ? args.pdf_path  // Pass URLs directly - backend handles downloading
      : resolve(args.pdf_path)

    const cmd = ["run", "--directory", TOOL_DIR, "pdf_ocr_backend.py", pdfPath]
    if (args.page) {
      cmd.push("--page", args.page)
    }

    try {
      const { stdout } = await execFileAsync("uv", cmd, { maxBuffer: 64 * 1024 * 1024 })
      return stdout.trim()
    } catch (error) {
      const stderr = error.stderr?.toString().trim()
      const stdout = error.stdout?.toString().trim()
      // On a partial page failure the backend prints the pages that worked to
      // stdout, so keep them rather than dropping real OCR output.
      const message =
        stderr && stdout
          ? `${stderr}\n\nPartial output:\n${stdout}`
          : stderr || stdout || `Command failed with exit code ${error.code}`
      throw new Error(message)
    }
  }
})