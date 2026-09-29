import { tool } from "@opencode-ai/plugin"
import { resolve, isAbsolute } from "path"

export default tool({
  description: "Extract text from PDF files using DeepSeek-OCR. Processes entire PDFs or specific pages and returns markdown or plain text output. Use this when you need to transcribe PDF documents for analysis or processing.",
  args: {
    pdf_path: tool.schema.string().describe("Path to PDF file (absolute or relative to current directory) or a URL to download the PDF from"),
    output_format: tool.schema.string().describe("Output format: 'markdown' or 'text' (defaults to 'markdown')"),
    page: tool.schema.string().optional().describe("Page(s) to OCR: single page ('5'), range ('1-5'), multiple ('1,3,5'), or mixed ('1-3,5,7-9'). If omitted, all pages are processed.")
  },
  async execute(args) {
    // Handle both local paths and URLs
    const pdfPath = args.pdf_path.startsWith('http://') || args.pdf_path.startsWith('https://')
      ? args.pdf_path  // Pass URLs directly - backend handles downloading
      : resolve(args.pdf_path)

    try {
      const result = args.page 
        ? await Bun.$`uv run --directory ~/.config/opencode/tool --env-file ~/.config/opencode/tool/.env pdf_ocr_backend.py ${pdfPath} ${args.output_format || 'markdown'} --page ${args.page}`.text()
        : await Bun.$`uv run --directory ~/.config/opencode/tool --env-file ~/.config/opencode/tool/.env pdf_ocr_backend.py ${pdfPath} ${args.output_format || 'markdown'}`.text()
      return result.trim()
    } catch (error) {
      if (error.exitCode !== 0) {
        const errorMsg = error.stdout?.toString().trim() || error.stderr?.toString().trim() || `Command failed with exit code ${error.exitCode}`
        throw new Error(errorMsg)
      }
      throw error
    }
  }
})
