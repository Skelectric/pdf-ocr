/**
 * pi pdf-ocr extension — registers a `pdf_ocr` tool that shells out to the
 * Python backend deployed at $PDF_OCR_TOOL_DIR (default ~/.config/pi/tool).
 *
 * The backend (pdf_ocr_backend.py) is harness-agnostic: it converts PDF pages
 * to images, sends them to the OCR route named in its route file, and prints
 * markdown. This extension is just the thin pi glue: argument marshalling +
 * invocation.
 *
 * Install: run ./deploy-tool.sh in the pdf-ocr repo (deploys the backend
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
// Above the backend's own 3600s request timeout, so the backend's actionable
// timeout message reaches the caller instead of a kill at the same moment.
const EXEC_TIMEOUT = 3660 * 1000;

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "pdf_ocr",
    label: "PDF OCR",
    description:
      "Extract text from PDF files using OCR. Sends the pages to the OCR route configured in the route file. Returns markdown. Accepts a local path or a http(s) URL to download.",
    promptSnippet: "OCR a PDF (local path or URL) to markdown",
    promptGuidelines: [
      "Use pdf_ocr when the user asks to transcribe or extract text from a PDF document.",
    ],
    parameters: Type.Object({
      pdf_path: Type.String({
        description:
          "Path to PDF (absolute, relative to cwd, or a http(s) URL to download)",
      }),
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
          `pdf_ocr backend not found at ${BACKEND}. Run ./deploy-tool.sh in the pdf-ocr repo (or set PDF_OCR_TOOL_DIR).`,
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
      // actionable message to stderr. On a partial page failure it also prints
      // the pages that worked to stdout, so keep those rather than dropping
      // real OCR output. Throwing (not returning isError) is how pi marks a
      // tool result as failed.
      const stderr = res.stderr?.trim();
      const stdout = res.stdout?.trim();
      const detail =
        stderr && stdout ? `${stderr}\n\nPartial output:\n${stdout}` : stderr || stdout;
      throw new Error(detail || `pdf_ocr failed (exit ${res.code})`);
    },
  });
}