/**
 * pi pdf-ocr extension — registers a `pdf_ocr` tool that shells out to the
 * Python backend deployed at $PDF_OCR_TOOL_DIR (default ~/.config/pi/tool).
 *
 * The backend (pdf_ocr_backend.py) is harness-agnostic: it converts PDF pages
 * to images, routes to an OCR model (DeepSeek-OCR / GLM-OCR / the currently
 * loaded multimodal model) via the llama-swap proxy, and prints markdown/text.
 * This extension is just the thin pi glue: argument marshalling + invocation.
 *
 * Install: run ./deploy-tool.sh in the opencode-ocr repo (deploys the backend
 * to ~/.config/pi/tool/ and this extension to ~/.pi/agent/extensions/pdf-ocr/),
 * then /reload in pi.
 */
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const TOOL_DIR =
  process.env.PDF_OCR_TOOL_DIR || path.join(os.homedir(), ".config", "pi", "tool");
const BACKEND = path.join(TOOL_DIR, "pdf_ocr_backend.py");
// Matches the 3600s client timeout the backend uses for upstream OCR calls.
const EXEC_TIMEOUT = 60 * 60 * 1000;

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "pdf_ocr",
    label: "PDF OCR",
    description:
      "Extract text from PDF files using OCR (DeepSeek-OCR / GLM-OCR / current multimodal model, routed via llama-swap). Returns markdown or plain text. Accepts a local path or a http(s) URL to download.",
    promptSnippet: "OCR a PDF (local path or URL) to markdown/text",
    promptGuidelines: [
      "Use pdf_ocr when the user asks to transcribe or extract text from a PDF document.",
    ],
    parameters: Type.Object({
      pdf_path: Type.String({
        description:
          "Path to PDF (absolute, relative to cwd, or a http(s) URL to download)",
      }),
      output_format: Type.Optional(
        Type.String({
          description:
            "Output format: 'markdown' or 'text' (defaults to 'markdown')",
        }),
      ),
      page: Type.Optional(
        Type.String({
          description:
            "Pages to OCR: '5', '1-5', '1,3,5', or '1-3,5,7-9'. Omit = all pages.",
        }),
      ),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      if (!fs.existsSync(BACKEND)) {
        throw new Error(
          `pdf_ocr backend not found at ${BACKEND}. Run ./deploy-tool.sh in the opencode-ocr repo (or set PDF_OCR_TOOL_DIR).`,
        );
      }

      const isUrl = /^https?:\/\//i.test(params.pdf_path);
      // Resolve relative paths against pi's cwd (not process.cwd()); URLs pass
      // through and are downloaded by the backend.
      const pdfPath = isUrl ? params.pdf_path : path.resolve(ctx.cwd, params.pdf_path);

      const args = [
        "run",
        "--directory",
        TOOL_DIR,
        BACKEND,
        pdfPath,
        params.output_format || "markdown",
      ];
      if (params.page) args.push("--page", params.page);

      onUpdate?.({
        content: [{ type: "text", text: `Running OCR on ${params.pdf_path}…` }],
      });

      const res = await pi.exec("uv", args, { signal, timeout: EXEC_TIMEOUT });

      if (res.code === 0) {
        return {
          content: [{ type: "text", text: res.stdout.trim() }],
          details: { exitCode: 0 },
        };
      }

      // exit 1 = general error, exit 3 = NO_OCR_SUPPORT; the backend prints an
      // actionable message to stderr. Throwing (not returning isError) is how
      // pi marks a tool result as failed.
      const msg =
        res.stderr?.trim() || res.stdout?.trim() || `pdf_ocr failed (exit ${res.code})`;
      throw new Error(msg);
    },
  });
}