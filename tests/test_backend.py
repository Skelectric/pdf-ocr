#!/usr/bin/env python3
import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tool"))
import pdf_ocr_backend
import pymupdf

from tests.support import TempDirTestCase, download_response, make_pdf, write_route


class TestMain(TempDirTestCase):
    """main() with the route file and the command-line overrides."""

    def setUp(self):
        super().setUp()
        self.test_pdf = make_pdf(self.temp_dir)
        self.route_file = write_route(
            self.temp_dir, {"model": "test-model", "base_url": "http://test:8080"}
        )

    def _argv(self, *extra):
        return [
            "pdf_ocr_backend.py",
            self.test_pdf,
            "--routing-config",
            self.route_file,
            *extra,
        ]

    @patch("pdf_ocr_backend.process_with_model")
    def test_main_prints_the_result(self, mock_process):
        mock_process.return_value = "Extracted text"

        with patch("sys.argv", self._argv()):
            with patch("builtins.print") as mock_print:
                pdf_ocr_backend.main()

        self.assertTrue(any("Extracted text" in str(c) for c in mock_print.call_args_list))

    @patch("pdf_ocr_backend.process_with_model")
    def test_main_uses_the_route_file(self, mock_process):
        mock_process.return_value = "text"

        with patch("sys.argv", self._argv()):
            with patch("builtins.print"):
                pdf_ocr_backend.main()

        args = mock_process.call_args[0]
        self.assertEqual(args[1], "test-model")
        self.assertEqual(args[2], "http://test:8080")

    @patch("pdf_ocr_backend.process_with_model")
    def test_main_route_flag_overrides_the_file(self, mock_process):
        mock_process.return_value = "text"

        with patch("sys.argv", self._argv("--route", "adhoc@http://adhoc:9000")):
            with patch("builtins.print"):
                pdf_ocr_backend.main()

        args = mock_process.call_args[0]
        self.assertEqual(args[1], "adhoc")
        self.assertEqual(args[2], "http://adhoc:9000")

    @patch("pdf_ocr_backend.process_with_model")
    def test_main_model_and_base_url_flags_override_the_file(self, mock_process):
        mock_process.return_value = "text"

        with patch(
            "sys.argv",
            self._argv("--model", "other-model", "--base-url", "http://other:9000"),
        ):
            with patch("builtins.print"):
                pdf_ocr_backend.main()

        args = mock_process.call_args[0]
        self.assertEqual(args[1], "other-model")
        self.assertEqual(args[2], "http://other:9000")

    @patch("builtins.print")
    def test_main_with_nonexistent_pdf(self, mock_print):
        with patch("sys.argv", ["pdf_ocr_backend.py", "/nonexistent/file.pdf"]):
            with self.assertRaises(SystemExit) as cm:
                pdf_ocr_backend.main()
            self.assertEqual(cm.exception.code, 1)

    @patch("builtins.print")
    def test_main_without_a_route_exits_1(self, mock_print):
        """No route file and no overrides is a configuration error."""
        missing = os.path.join(self.temp_dir, "missing.json")

        with patch(
            "sys.argv", ["pdf_ocr_backend.py", self.test_pdf, "--routing-config", missing]
        ):
            with self.assertRaises(SystemExit) as cm:
                pdf_ocr_backend.main()

        self.assertEqual(cm.exception.code, 1)
        self.assertTrue(
            any("no OCR route configured" in str(c) for c in mock_print.call_args_list)
        )

    @patch("builtins.print")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_main_with_corrupt_pdf(self, mock_pymupdf_open, mock_print):
        mock_pymupdf_open.side_effect = Exception("Corrupt PDF")

        with patch("sys.argv", self._argv()):
            with self.assertRaises(SystemExit) as cm:
                pdf_ocr_backend.main()
            self.assertEqual(cm.exception.code, 1)

    @patch("pdf_ocr_backend.OpenAI")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_main_with_a_dead_endpoint_exits_1(self, mock_pymupdf_open, mock_openai):
        """Every page failing must not look like success."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_pymupdf_open.return_value = mock_doc
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("unavailable")
        mock_openai.return_value = mock_client

        with patch("sys.argv", self._argv()):
            with patch("builtins.print"):
                with self.assertRaises(SystemExit) as cm:
                    pdf_ocr_backend.main()

        self.assertEqual(cm.exception.code, 1)


class TestRouteResolution(TempDirTestCase):
    """The route file and the command-line overrides decide the route."""

    def setUp(self):
        super().setUp()
        self.route_file = os.path.join(self.temp_dir, "ocr_routing.json")

    def _write_route(self, text):
        with open(self.route_file, "w") as f:
            f.write(text)

    def test_resolve_route_reads_the_file(self):
        self._write_route(json.dumps({"model": "m", "base_url": "http://h:1"}))
        self.assertEqual(
            pdf_ocr_backend.resolve_route(Path(self.route_file)),
            ("m", "http://h:1", None),
        )

    def test_resolve_route_carries_the_prompt_family(self):
        self._write_route(
            json.dumps({"model": "m", "base_url": "http://h:1", "prompt": "generic"})
        )
        self.assertEqual(
            pdf_ocr_backend.resolve_route(Path(self.route_file)),
            ("m", "http://h:1", "generic"),
        )

    def test_resolve_route_overrides(self):
        self._write_route(json.dumps({"model": "m", "base_url": "http://h:1"}))
        self.assertEqual(
            pdf_ocr_backend.resolve_route(
                Path(self.route_file),
                model="o",
                base_url="http://o:2/",
                prompt="deepseek-ocr",
            ),
            ("o", "http://o:2", "deepseek-ocr"),
        )

    def test_resolve_route_needs_a_model(self):
        self._write_route(json.dumps({"base_url": "http://h:1"}))
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.resolve_route(Path(self.route_file))

    def test_resolve_route_needs_a_base_url(self):
        self._write_route(json.dumps({"model": "m"}))
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.resolve_route(Path(self.route_file))

    def test_missing_route_file(self):
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.resolve_route(Path(self.route_file))

    def test_invalid_json(self):
        self._write_route("{not json")
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.resolve_route(Path(self.route_file))

    def test_route_file_must_be_an_object(self):
        self._write_route('["nope"]')
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.resolve_route(Path(self.route_file))

    def test_parse_route_spec_sets_both(self):
        self.assertEqual(
            pdf_ocr_backend.parse_route_spec("m@http://h:1"), ("m", "http://h:1")
        )

    def test_parse_route_spec_model_only(self):
        self.assertEqual(pdf_ocr_backend.parse_route_spec("m"), ("m", None))

    def test_parse_route_spec_empty(self):
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.parse_route_spec("   ")

    def test_no_discovery_helpers_remain(self):
        """The tool must not scan the endpoint for loaded models."""
        for name in (
            "get_all_models",
            "get_current_model",
            "check_multimodal_support",
            "get_ocr_method_for_model",
            "get_ocr_method_for_model_set",
            "load_ocr_routing_config",
            "resolve_ocr_route",
        ):
            self.assertFalse(hasattr(pdf_ocr_backend, name), name)


class TestPromptFamily(unittest.TestCase):
    """The route's prompt family selects the template."""

    def test_family_wins_over_the_model_id(self):
        template = pdf_ocr_backend.get_prompt_template(
            "some-pool-label", "deepseek-ocr"
        )
        self.assertIn("skip_special_tokens", template["extra_body"])

    def test_generic_alias(self):
        template = pdf_ocr_backend.get_prompt_template("deepseek-ocr-2", "generic")
        self.assertIsNone(template["extra_body"])

    def test_model_id_fallback(self):
        self.assertIn(
            "skip_special_tokens",
            pdf_ocr_backend.get_prompt_template("deepseek-ocr-2")["extra_body"],
        )
        self.assertIsNone(pdf_ocr_backend.get_prompt_template("qwen-vl")["extra_body"])


class TestByteFallbackDecoding(unittest.TestCase):
    """The endpoint's byte-fallback markers are decoded for every family."""

    def _ocr_with(self, template):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = "a\u0120b\u010ac"
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        page = MagicMock()
        page.get_pixmap.return_value.tobytes.return_value = b"png"
        return pdf_ocr_backend._ocr_page(client, page, "m", template)

    def test_the_generic_family_decodes(self):
        template = pdf_ocr_backend.get_prompt_template("ocr-model-pool", "generic")
        self.assertEqual(self._ocr_with(template), "a b\nc")

    def test_the_deepseek_family_decodes(self):
        template = pdf_ocr_backend.get_prompt_template("deepseek-ocr-2")
        self.assertEqual(self._ocr_with(template), "a b\nc")

    def test_unknown_family(self):
        with self.assertRaises(pdf_ocr_backend.ConfigurationError):
            pdf_ocr_backend.get_prompt_template("m", "nope")


class TestExitCode3(TempDirTestCase):
    """Exit 3: the endpoint rejects the configured model for OCR."""

    def setUp(self):
        super().setUp()
        self.test_pdf = make_pdf(self.temp_dir)

    def test_endpoint_rejection_raises_no_ocr_support(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = pdf_ocr_backend.BadRequestError(
            "model does not support images", response=MagicMock(), body=None
        )
        page = MagicMock()
        page.get_pixmap.return_value.tobytes.return_value = b"png"

        with self.assertRaises(pdf_ocr_backend.NoOcrSupportError) as cm:
            pdf_ocr_backend._ocr_page(client, page, "text-model", {"user": "u"})

        self.assertIn("text-model", str(cm.exception))

    @patch("pdf_ocr_backend._ocr_page")
    def test_a_route_rejection_aborts_the_whole_run(self, mock_page):
        mock_page.side_effect = pdf_ocr_backend.NoOcrSupportError("NO_OCR_SUPPORT: nope")
        doc = pymupdf.open()
        doc.new_page(width=612, height=792)
        doc.new_page(width=612, height=792)
        try:
            with self.assertRaises(pdf_ocr_backend.NoOcrSupportError):
                pdf_ocr_backend.process_pdf_pages(doc, "m", "http://h:1", {"user": "u"})
        finally:
            doc.close()

    @patch("pdf_ocr_backend._ocr_page")
    def test_main_maps_no_ocr_support_to_exit_3(self, mock_page):
        mock_page.side_effect = pdf_ocr_backend.NoOcrSupportError("NO_OCR_SUPPORT: nope")
        route_file = os.path.join(self.temp_dir, "ocr_routing.json")
        with open(route_file, "w") as f:
            json.dump({"model": "m", "base_url": "http://h:1"}, f)

        with patch(
            "sys.argv", ["pdf_ocr_backend.py", self.test_pdf, "--routing-config", route_file]
        ):
            with patch.object(Path, "exists", return_value=True):
                with patch("sys.stderr") as mock_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        pdf_ocr_backend.main()

        self.assertEqual(cm.exception.code, 3)
        error_output = "".join(
            str(call[0][0]) for call in mock_stderr.write.call_args_list if call[0]
        )
        self.assertIn("NO_OCR_SUPPORT", error_output)


class TestURLHandling(unittest.TestCase):
    """Unit tests for URL detection and downloading."""

    def test_is_url_http(self):
        """Test http:// is detected as URL."""
        self.assertTrue(pdf_ocr_backend.is_url("http://example.com/test.pdf"))

    def test_is_url_https(self):
        """Test https:// is detected as URL."""
        self.assertTrue(pdf_ocr_backend.is_url("https://example.com/test.pdf"))

    def test_is_url_local_path(self):
        """Test local path is not detected as URL."""
        self.assertFalse(pdf_ocr_backend.is_url("/path/to/file.pdf"))

    def test_is_url_relative_path(self):
        """Test relative path is not detected as URL."""
        self.assertFalse(pdf_ocr_backend.is_url("./file.pdf"))

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_download_pdf_success(self, mock_pymupdf_open, mock_get):
        """Test successful PDF download."""
        mock_get.return_value = download_response(
            {"Content-Length": "1024"}, [b"fake pdf content"]
        )

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_pymupdf_open.return_value = mock_doc

        temp_path = pdf_ocr_backend.download_pdf("https://example.com/test.pdf")

        self.assertTrue(os.path.exists(temp_path))
        self.assertTrue(temp_path.endswith(".pdf"))
        mock_get.assert_called_once_with(
            "https://example.com/test.pdf", stream=True, timeout=60, allow_redirects=True
        )

        # Cleanup
        pdf_ocr_backend.cleanup_temp_file(temp_path)

    @patch("pdf_ocr_backend.requests.get")
    def test_download_pdf_http_error(self, mock_get):
        """Test download failure on HTTP error."""
        import requests

        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with self.assertRaises(pdf_ocr_backend.DownloadError) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/missing.pdf")

        # The message names the URL that failed, which is the actionable part.
        self.assertIn("https://example.com/missing.pdf", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    def test_download_pdf_connection_error(self, mock_get):
        """Test download failure on connection error."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(pdf_ocr_backend.DownloadError):
            pdf_ocr_backend.download_pdf("https://example.com/test.pdf")

    @patch("pdf_ocr_backend.requests.get")
    def test_download_pdf_content_length_exceeded(self, mock_get):
        """Test download rejected when Content-Length exceeds limit."""
        # 100 MB
        mock_get.return_value = download_response(
            {"Content-Length": str(100 * 1024 * 1024)}
        )

        with self.assertRaises(pdf_ocr_backend.DownloadError) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/huge.pdf")

        # The message states the limit, which is the actionable part.
        self.assertIn("50 MB", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_download_pdf_streaming_size_exceeded(self, mock_pymupdf_open, mock_get):
        """Test download rejected when streaming exceeds limit."""
        # Simulate chunks that exceed 50MB
        mock_get.return_value = download_response(chunks=[b"x" * (1024 * 1024)] * 60)

        with self.assertRaises(pdf_ocr_backend.DownloadError):
            pdf_ocr_backend.download_pdf("https://example.com/huge.pdf")

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_download_pdf_invalid_content(self, mock_pymupdf_open, mock_get):
        """Test download rejected when content is not a valid PDF."""
        mock_get.return_value = download_response(
            {"Content-Length": "100"}, [b"not a pdf"]
        )

        mock_pymupdf_open.side_effect = Exception("cannot open document")

        with self.assertRaises(pdf_ocr_backend.DownloadError):
            pdf_ocr_backend.download_pdf("https://example.com/notpdf.txt")

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_download_pdf_empty_file(self, mock_pymupdf_open, mock_get):
        """Test download rejected when file is empty."""
        mock_get.return_value = download_response({"Content-Length": "0"})

        with self.assertRaises(pdf_ocr_backend.DownloadError):
            pdf_ocr_backend.download_pdf("https://example.com/empty.pdf")

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.pymupdf.open")
    def test_download_pdf_no_pages(self, mock_pymupdf_open, mock_get):
        """Test download rejected when PDF has no pages."""
        mock_get.return_value = download_response(
            {"Content-Length": "1024"}, [b"fake pdf content"]
        )

        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_pymupdf_open.return_value = mock_doc

        with self.assertRaises(pdf_ocr_backend.DownloadError):
            pdf_ocr_backend.download_pdf("https://example.com/empty.pdf")

    def test_cleanup_temp_file_success(self):
        """Test cleanup removes existing file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name

        self.assertTrue(os.path.exists(temp_path))
        pdf_ocr_backend.cleanup_temp_file(temp_path)
        self.assertFalse(os.path.exists(temp_path))

    def test_cleanup_temp_file_nonexistent(self):
        """Test cleanup handles non-existent file gracefully."""
        temp_path = "/tmp/nonexistent_cleanup_test.pdf"
        # Should not raise
        pdf_ocr_backend.cleanup_temp_file(temp_path)

    @patch("pdf_ocr_backend.download_pdf")
    @patch("pdf_ocr_backend.route_ocr_request")
    def test_main_with_url_success(self, mock_route, mock_download):
        """Test main() with a valid URL downloads and processes PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name

        mock_download.return_value = temp_path
        mock_route.return_value = "OCR result from URL"

        with patch("sys.argv", ["pdf_ocr_backend.py", "https://example.com/test.pdf"]):
            with patch("builtins.print") as mock_print:
                pdf_ocr_backend.main()

        mock_download.assert_called_once_with("https://example.com/test.pdf")
        mock_route.assert_called_once_with(
            temp_path,
            None,
            model=None,
            base_url=None,
            prompt=None,
            routing_config_path=None,
        )
        calls = mock_print.call_args_list
        self.assertTrue(any("OCR result from URL" in str(call) for call in calls))

        # Cleanup if file still exists
        pdf_ocr_backend.cleanup_temp_file(temp_path)

    @patch("pdf_ocr_backend.download_pdf")
    def test_main_with_url_download_failure(self, mock_download):
        """Test main() exits with code 1 when URL download fails."""
        mock_download.side_effect = pdf_ocr_backend.DownloadError("Download failed")

        with patch("sys.argv", ["pdf_ocr_backend.py", "https://example.com/bad.pdf"]):
            with patch("builtins.print") as mock_print:
                with self.assertRaises(SystemExit) as cm:
                    pdf_ocr_backend.main()
                self.assertEqual(cm.exception.code, 1)

        # The error goes to stderr, so stdout stays the OCR channel.
        calls = mock_print.call_args_list
        self.assertTrue(
            any(
                "Download failed" in str(call)
                and call.kwargs.get("file") is sys.stderr
                for call in calls
            )
        )


class TestRouteOverrides(TempDirTestCase):
    """Command-line overrides and the route-file path precedence."""

    def setUp(self):
        super().setUp()
        self.test_pdf = make_pdf(self.temp_dir)
        self.route_file = write_route(
            self.temp_dir, {"model": "file-model", "base_url": "http://file:8080"}
        )

    @patch("pdf_ocr_backend.process_with_model")
    def test_explicit_base_url_overrides_the_file(self, mock_process):
        mock_process.return_value = "ok"

        result = pdf_ocr_backend.route_ocr_request(
            self.test_pdf,
            base_url="http://explicit:8080",
            routing_config_path=self.route_file,
        )

        self.assertEqual(result, "ok")
        mock_process.assert_called_once_with(
            self.test_pdf, "file-model", "http://explicit:8080", None, None
        )

    @patch("pdf_ocr_backend.process_with_model")
    def test_explicit_model_and_prompt_override_the_file(self, mock_process):
        mock_process.return_value = "ok"

        pdf_ocr_backend.route_ocr_request(
            self.test_pdf,
            model="m2",
            prompt="generic",
            routing_config_path=self.route_file,
        )

        mock_process.assert_called_once_with(
            self.test_pdf, "m2", "http://file:8080", None, "generic"
        )

    def test_route_file_path_is_used(self):
        self.assertEqual(
            pdf_ocr_backend.resolve_routing_config_path(Path(self.route_file)),
            Path(self.route_file),
        )

    def test_resolve_routing_config_path_precedence(self):
        """Explicit arg > PDF_OCR_ROUTING_CONFIG env > tool-dir default."""
        explicit = Path("/tmp/explicit.json")
        self.assertEqual(pdf_ocr_backend.resolve_routing_config_path(explicit), explicit)

        with patch.dict(os.environ, {"PDF_OCR_ROUTING_CONFIG": "/tmp/env.json"}):
            self.assertEqual(
                pdf_ocr_backend.resolve_routing_config_path(), Path("/tmp/env.json")
            )

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                pdf_ocr_backend.resolve_routing_config_path(),
                Path(pdf_ocr_backend.__file__).parent / "ocr_routing.json",
            )

    def test_load_route_honors_the_env_path(self):
        route_content = json.dumps({"model": "m", "base_url": "http://h:1"})

        with patch.dict(os.environ, {"PDF_OCR_ROUTING_CONFIG": "/tmp/env-routing.json"}):
            with patch("builtins.open", mock_open(read_data=route_content)) as m:
                with patch.object(Path, "exists", return_value=True):
                    route = pdf_ocr_backend.load_route()

        self.assertEqual(route["model"], "m")
        m.assert_called_once_with(Path("/tmp/env-routing.json"), "r")


class TestPageSpecParsing(unittest.TestCase):
    """parse_page_spec accepts the documented forms and rejects bad input."""

    def test_single_page(self):
        self.assertEqual(pdf_ocr_backend.parse_page_spec("3", 10), [2])

    def test_range(self):
        self.assertEqual(pdf_ocr_backend.parse_page_spec("1-5", 10), [0, 1, 2, 3, 4])

    def test_mixed(self):
        self.assertEqual(
            pdf_ocr_backend.parse_page_spec("1-3,5,7-9", 10), [0, 1, 2, 4, 6, 7, 8]
        )

    def test_none_means_all_pages(self):
        self.assertEqual(pdf_ocr_backend.parse_page_spec(None, 3), [0, 1, 2])

    def test_negative_page_rejected(self):
        with self.assertRaises(ValueError):
            pdf_ocr_backend.parse_page_spec("-5", 10)

    def test_double_dash_rejected(self):
        with self.assertRaises(ValueError):
            pdf_ocr_backend.parse_page_spec("1--2", 10)

    def test_page_beyond_document_rejected(self):
        with self.assertRaises(ValueError):
            pdf_ocr_backend.parse_page_spec("11", 10)

    def test_start_after_end_rejected(self):
        with self.assertRaises(ValueError):
            pdf_ocr_backend.parse_page_spec("5-2", 10)


class TestPageFailures(unittest.TestCase):
    """One failing page must not lose the pages that succeeded."""

    def _doc(self):
        doc = pymupdf.open()
        doc.new_page(width=612, height=792)
        doc.new_page(width=612, height=792)
        return doc

    @patch("pdf_ocr_backend._ocr_page")
    def test_failed_page_is_marked_and_others_survive(self, mock_page):
        """The good pages are kept, the bad one is marked, and the run fails."""
        mock_page.side_effect = [Exception("boom"), "second page text"]
        doc = self._doc()
        try:
            with self.assertRaises(pdf_ocr_backend.PageFailureError) as cm:
                pdf_ocr_backend.process_pdf_pages(
                    doc, "m", "http://test:8080", {"user": "u"}
                )
        finally:
            doc.close()

        results = cm.exception.results
        self.assertEqual(len(results), 2)
        self.assertIn("[OCR failed: boom]", results[0])
        self.assertIn("second page text", results[1])

    @patch("pdf_ocr_backend._ocr_page")
    def test_all_pages_succeeding_returns_them(self, mock_page):
        mock_page.side_effect = ["one", "two"]
        doc = self._doc()
        try:
            results = pdf_ocr_backend.process_pdf_pages(
                doc, "m", "http://test:8080", {"user": "u"}
            )
        finally:
            doc.close()

        self.assertEqual(len(results), 2)
        self.assertIn("one", results[0])

    def test_none_content_raises_instead_of_rendering_none(self):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        page = MagicMock()
        page.get_pixmap.return_value.tobytes.return_value = b"png"

        with self.assertRaises(pdf_ocr_backend.OcrRequestError):
            pdf_ocr_backend._ocr_page(client, page, "m", {"user": "u"})


if __name__ == "__main__":
    unittest.main()
