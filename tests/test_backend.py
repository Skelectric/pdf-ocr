#!/usr/bin/env python3
import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pdf-ocr" / "tool"))
import pdf_ocr_backend
import fitz


class TestOCRRoutingConfig(unittest.TestCase):
    """Unit tests for OCR routing configuration functionality."""

    def test_load_ocr_routing_config_existing_valid(self):
        """Test loading existing valid config file."""
        config_content = json.dumps(
            {"ocr_routing": {"test-model": "deepseek-ocr"}, "default": "current_model"}
        )

        with patch("builtins.open", mock_open(read_data=config_content)):
            with patch.object(Path, "exists", return_value=True):
                result = pdf_ocr_backend.load_ocr_routing_config(Path("/fake/path"))

        self.assertEqual(result["ocr_routing"]["test-model"], "deepseek-ocr")
        self.assertEqual(result["default"], "current_model")

    def test_load_ocr_routing_config_nonexistent(self):
        """Test loading non-existent config file returns defaults."""
        with patch.object(Path, "exists", return_value=False):
            result = pdf_ocr_backend.load_ocr_routing_config(Path("/fake/path"))

        self.assertEqual(result["ocr_routing"], {})
        self.assertEqual(result["default"], "current_model")

    def test_load_ocr_routing_config_invalid_json(self):
        """Test loading invalid JSON returns defaults."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch.object(Path, "exists", return_value=True):
                result = pdf_ocr_backend.load_ocr_routing_config(Path("/fake/path"))

        self.assertEqual(result["ocr_routing"], {})
        self.assertEqual(result["default"], "current_model")

    def test_load_ocr_routing_config_missing_ocr_routing_key(self):
        """Test config with missing ocr_routing key gets default."""
        config_content = json.dumps({"default": "deepseek-ocr"})

        with patch("builtins.open", mock_open(read_data=config_content)):
            with patch.object(Path, "exists", return_value=True):
                result = pdf_ocr_backend.load_ocr_routing_config(Path("/fake/path"))

        self.assertEqual(result["ocr_routing"], {})
        self.assertEqual(result["default"], "deepseek-ocr")

    def test_load_ocr_routing_config_missing_default_key(self):
        """Test config with missing default key gets default."""
        config_content = json.dumps({"ocr_routing": {"model": "deepseek-ocr"}})

        with patch("builtins.open", mock_open(read_data=config_content)):
            with patch.object(Path, "exists", return_value=True):
                result = pdf_ocr_backend.load_ocr_routing_config(Path("/fake/path"))

        self.assertEqual(result["default"], "current_model")

    def test_get_ocr_method_for_model_exact_match(self):
        """Test exact model match in routing config."""
        config = {
            "ocr_routing": {"ik_llama.cpp/kimi-k2.5": "deepseek-ocr"},
            "default": "current_model",
        }

        result = pdf_ocr_backend.get_ocr_method_for_model(
            "ik_llama.cpp/kimi-k2.5", config
        )
        self.assertEqual(result, "deepseek-ocr")

    def test_get_ocr_method_for_model_partial_match_pattern_in_model(self):
        """Test partial match where pattern is contained in model_id."""
        config = {
            "ocr_routing": {"kimi-k2.5": "deepseek-ocr"},
            "default": "current_model",
        }

        result = pdf_ocr_backend.get_ocr_method_for_model(
            "ik_llama.cpp/kimi-k2.5-experimental", config
        )
        self.assertEqual(result, "deepseek-ocr")

    def test_get_ocr_method_for_model_partial_match_model_in_pattern(self):
        """Test partial match where model_id is contained in pattern."""
        config = {
            "ocr_routing": {"ik_llama.cpp/kimi-k2.5-experimental": "deepseek-ocr"},
            "default": "current_model",
        }

        result = pdf_ocr_backend.get_ocr_method_for_model("kimi-k2.5", config)
        self.assertEqual(result, "deepseek-ocr")

    def test_get_ocr_method_for_model_no_match_returns_default(self):
        """Test no match returns default method."""
        config = {
            "ocr_routing": {"other-model": "deepseek-ocr"},
            "default": "current_model",
        }

        result = pdf_ocr_backend.get_ocr_method_for_model("unknown-model", config)
        self.assertEqual(result, "current_model")

    def test_get_ocr_method_for_model_empty_config(self):
        """Test empty config returns default method."""
        config = {"ocr_routing": {}, "default": "current_model"}

        result = pdf_ocr_backend.get_ocr_method_for_model("any-model", config)
        self.assertEqual(result, "current_model")


class TestModelSetRouting(unittest.TestCase):
    """Unit tests for model-set based OCR routing."""

    def test_exact_model_set_match(self):
        """Exact model-set key matches loaded models."""
        config = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
                "kimi-k2.6": "deepseek-ocr-2"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["kimi-k2.6", "qwen3.6-27b-nvfp4"], config
        )
        self.assertEqual(result, "glm-ocr")  # Exact set match wins

    def test_model_set_no_match_fallback_to_single(self):
        """No exact set match falls back to single-model matching."""
        config = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
                "kimi-k2.6": "deepseek-ocr-2"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["kimi-k2.6", "qwen3.6-35b-a3b-nvfp4"], config
        )
        self.assertEqual(result, "deepseek-ocr-2")  # Single-model match

    def test_model_set_order_independent(self):
        """Model-set matching is order-independent."""
        config = {
            "ocr_routing": {
                "qwen3.6-27b-nvfp4,kimi-k2.6": "glm-ocr"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["kimi-k2.6", "qwen3.6-27b-nvfp4"], config
        )
        self.assertEqual(result, "glm-ocr")

    def test_single_model_no_set_match(self):
        """Single model loaded, no set key matches."""
        config = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
                "kimi-k2.6": "deepseek-ocr-2"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(["kimi-k2.6"], config)
        self.assertEqual(result, "deepseek-ocr-2")

    def test_no_match_returns_default(self):
        """No matching keys returns default."""
        config = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(["unknown-model"], config)
        self.assertEqual(result, "current_model")

    def test_case_insensitive_matching(self):
        """Model IDs are matched case-insensitively."""
        config = {
            "ocr_routing": {
                "KIMI-K2.6,QWEN3.6-27B-NVFP4": "glm-ocr"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["kimi-k2.6", "qwen3.6-27b-nvfp4"], config
        )
        self.assertEqual(result, "glm-ocr")


class TestGetAllModels(unittest.TestCase):
    """Unit tests for get_all_models function."""

    @patch("pdf_ocr_backend.requests.get")
    def test_get_all_models_proxy_format(self, mock_get):
        """Test get_all_models with the proxy /running format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "running": [
                {"model": "kimi-k2.6"},
                {"model": "qwen3.6-35b-a3b-nvfp4"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_all_models("http://test:8080")

        self.assertEqual(result, ["kimi-k2.6", "qwen3.6-35b-a3b-nvfp4"])

    @patch("pdf_ocr_backend.requests.get")
    def test_get_all_models_empty(self, mock_get):
        """Test get_all_models with no running models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"running": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_all_models("http://test:8080")

        self.assertEqual(result, [])

    @patch("pdf_ocr_backend.requests.get")
    def test_get_all_models_list_format(self, mock_get):
        """Test get_all_models with list response format."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"model": "model-a"},
            {"model": "model-b"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_all_models("http://test:8080")

        self.assertEqual(result, ["model-a", "model-b"])

    @patch("pdf_ocr_backend.requests.get")
    def test_get_all_models_connection_error(self, mock_get):
        """Test get_all_models with connection failure."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = pdf_ocr_backend.get_all_models("http://test:8080")

        self.assertEqual(result, [])


class TestModelDetection(unittest.TestCase):
    """Unit tests for model detection functionality."""

    @patch("pdf_ocr_backend.requests.get")
    def test_get_current_model_list_format(self, mock_get):
        """Test get_current_model with list response format."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"model": "test-model-v1", "id": "model-123"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_current_model("http://test:8080")

        self.assertEqual(result, "test-model-v1")
        mock_get.assert_called_once_with("http://test:8080/running", timeout=10)

    @patch("pdf_ocr_backend.requests.get")
    def test_get_current_model_dict_format(self, mock_get):
        """Test get_current_model with dict response format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test-model-v2",
            "status": "running",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_current_model("http://test:8080")

        self.assertEqual(result, "test-model-v2")

    @patch("pdf_ocr_backend.requests.get")
    def test_get_current_model_empty_list(self, mock_get):
        """Test get_current_model with empty list response."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_current_model("http://test:8080")

        self.assertIsNone(result)

    @patch("pdf_ocr_backend.requests.get")
    def test_get_current_model_connection_error(self, mock_get):
        """Test get_current_model with connection failure."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = pdf_ocr_backend.get_current_model("http://test:8080")

        self.assertIsNone(result)

    @patch("pdf_ocr_backend.requests.get")
    def test_get_current_model_timeout(self, mock_get):
        """Test get_current_model with timeout."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = pdf_ocr_backend.get_current_model("http://test:8080")

        self.assertIsNone(result)

    @patch("pdf_ocr_backend.requests.get")
    def test_check_multimodal_support_vision_true(self, mock_get):
        """Test check_multimodal_support with vision=true."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "modalities": {"vision": True, "audio": False}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.check_multimodal_support(
            "http://test:8080", "test-model"
        )

        self.assertTrue(result)
        mock_get.assert_called_once_with(
            "http://test:8080/upstream/test-model/props", timeout=10
        )

    @patch("pdf_ocr_backend.requests.get")
    def test_check_multimodal_support_vision_false(self, mock_get):
        """Test check_multimodal_support with vision=false."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "modalities": {"vision": False, "audio": True}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.check_multimodal_support(
            "http://test:8080", "test-model"
        )

        self.assertFalse(result)

    @patch("pdf_ocr_backend.requests.get")
    def test_check_multimodal_support_missing_modalities(self, mock_get):
        """Test check_multimodal_support with missing modalities field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "test-model", "name": "Test Model"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.check_multimodal_support(
            "http://test:8080", "test-model"
        )

        self.assertFalse(result)

    @patch("pdf_ocr_backend.requests.get")
    def test_check_multimodal_support_connection_error(self, mock_get):
        """Test check_multimodal_support with connection failure."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = pdf_ocr_backend.check_multimodal_support(
            "http://test:8080", "test-model"
        )

        self.assertFalse(result)


class TestModelRouter(unittest.TestCase):
    """Unit tests for model routing functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = os.path.join(self.temp_dir, "test.pdf")
        # Create a minimal valid PDF for fitz.open()
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(self.test_pdf)
        doc.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_get_prompt_template_deepseek_ocr(self):
        """Test get_prompt_template returns deepseek-ocr template."""
        template = pdf_ocr_backend.get_prompt_template("deepseek-ocr-v1")
        self.assertIsNone(template["system"])
        self.assertEqual(template["user"], "Free OCR.")
        self.assertIsNotNone(template["extra_body"])
        self.assertIn("skip_special_tokens", template["extra_body"])

    def test_get_prompt_template_default_multimodal(self):
        """Test get_prompt_template returns default_multimodal template."""
        template = pdf_ocr_backend.get_prompt_template("qwen-vl")
        self.assertIsNotNone(template["system"])
        self.assertIn("OCR assistant", template["system"])
        self.assertEqual(template["extra_body"], None)

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_deepseek_ocr_configured_success(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """Test routing to DeepSeek-OCR when configured and successful."""
        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.return_value = "DeepSeek OCR result"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "DeepSeek OCR result")
        mock_process.assert_called_once_with(
            self.test_pdf, "markdown", "deepseek-ocr", "http://test:8080/v1", [0]
        )

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_deepseek_ocr_configured_failure(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """Test routing to DeepSeek-OCR when configured but fails (Exit 1)."""
        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.side_effect = Exception("DeepSeek-OCR failed")

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertIn("DeepSeek-OCR failed", str(cm.exception))

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_current_model_with_vision(
        self, mock_get_models, mock_load_config, mock_multimodal, mock_process
    ):
        """Test routing to current model with vision support (Exit 0)."""
        mock_get_models.return_value = ["qwen-vl"]
        mock_load_config.return_value = {
            "ocr_routing": {"qwen-vl": "current_model"},
            "default": "current_model",
        }
        mock_multimodal.return_value = True
        mock_process.return_value = "Current model OCR result"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "Current model OCR result")
        mock_process.assert_called_once_with(
            self.test_pdf, "markdown", "qwen-vl", "http://test:8080/v1", [0]
        )

    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_current_model_no_vision_exit_3(
        self, mock_get_models, mock_load_config, mock_multimodal
    ):
        """Test routing to current model without vision support (Exit 3)."""
        mock_get_models.return_value = ["text-only-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"text-only-model": "current_model"},
            "default": "current_model",
        }
        mock_multimodal.return_value = False

        with self.assertRaises(SystemExit) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(cm.exception.code, 3)

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_model_not_in_config_with_vision(
        self, mock_get_models, mock_load_config, mock_multimodal, mock_process
    ):
        """Test routing when model not in config but has vision support."""
        mock_get_models.return_value = ["new-model"]
        mock_load_config.return_value = {"ocr_routing": {}, "default": "current_model"}
        mock_multimodal.return_value = True
        mock_process.return_value = "OCR result"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "OCR result")

    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_model_not_in_config_no_vision_exit_3(
        self, mock_get_models, mock_load_config, mock_multimodal
    ):
        """Test routing when model not in config and no vision support (Exit 3)."""
        mock_get_models.return_value = ["text-model"]
        mock_load_config.return_value = {"ocr_routing": {}, "default": "current_model"}
        mock_multimodal.return_value = False

        with self.assertRaises(SystemExit) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(cm.exception.code, 3)

    @patch.dict(os.environ, {}, clear=True)
    def test_route_ocr_request_no_base_url(self):
        """Test route_ocr_request when DEEPSEEK_OCR_BASE_URL not set."""
        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertIn("DEEPSEEK_OCR_BASE_URL not set", str(cm.exception))

    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_no_model_loaded(self, mock_get_models):
        """Test route_ocr_request when no model is loaded."""
        mock_get_models.return_value = []

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertIn("No model currently loaded", str(cm.exception))


class TestPDFOCRBackend(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = os.path.join(self.temp_dir, "test.pdf")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch("pdf_ocr_backend.fitz.open")
    @patch("pdf_ocr_backend.OpenAI")
    def test_main_with_valid_pdf(
        self,
        mock_openai,
        mock_fitz_open,
        mock_get_models,
        mock_load_config,
        mock_process,
    ):
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"fake_png_bytes"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Extracted text"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.return_value = "Extracted text"

        with patch("sys.argv", ["pdf_ocr_backend.py", self.test_pdf, "markdown"]):
            with patch("builtins.print") as mock_print:
                with patch.object(Path, "exists", return_value=True):
                    with patch.dict(
                        os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"}
                    ):
                        pdf_ocr_backend.main()

        calls = mock_print.call_args_list
        self.assertTrue(any("Extracted text" in str(call) for call in calls))

    @patch("builtins.print")
    def test_main_with_nonexistent_pdf(self, mock_print):
        with patch(
            "sys.argv", ["pdf_ocr_backend.py", "/nonexistent/file.pdf", "markdown"]
        ):
            with self.assertRaises(SystemExit) as cm:
                pdf_ocr_backend.main()
            self.assertEqual(cm.exception.code, 1)

    @patch("builtins.print")
    @patch("pdf_ocr_backend.fitz.open")
    def test_main_with_corrupt_pdf(self, mock_fitz_open, mock_print):
        mock_fitz_open.side_effect = Exception("Corrupt PDF")

        with patch("sys.argv", ["pdf_ocr_backend.py", self.test_pdf, "markdown"]):
            with patch.object(Path, "exists", return_value=True):
                with self.assertRaises(SystemExit) as cm:
                    pdf_ocr_backend.main()
                self.assertEqual(cm.exception.code, 1)

    @patch("pdf_ocr_backend.fitz.open")
    @patch("pdf_ocr_backend.OpenAI")
    def test_main_with_ocr_error(self, mock_openai, mock_fitz_open):
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"fake_png_bytes"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "OCR service unavailable"
        )
        mock_openai.return_value = mock_client

        with patch("sys.argv", ["pdf_ocr_backend.py", self.test_pdf, "markdown"]):
            with patch.object(Path, "exists", return_value=True):
                with patch.dict(
                    os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"}
                ):
                    with self.assertRaises(SystemExit) as cm:
                        pdf_ocr_backend.main()
                    self.assertEqual(cm.exception.code, 1)

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch("pdf_ocr_backend.fitz.open")
    @patch("pdf_ocr_backend.OpenAI")
    def test_main_default_output_format(
        self,
        mock_openai,
        mock_fitz_open,
        mock_get_models,
        mock_load_config,
        mock_process,
    ):
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"fake_png_bytes"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Default format text"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.return_value = "Default format text"

        with patch("sys.argv", ["pdf_ocr_backend.py", self.test_pdf]):
            with patch("builtins.print"):
                with patch.object(Path, "exists", return_value=True):
                    with patch.dict(
                        os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"}
                    ):
                        pdf_ocr_backend.main()

        mock_process.assert_called_once()

    @patch("builtins.print")
    @patch("pdf_ocr_backend.fitz.open")
    def test_main_without_base_url(self, mock_fitz_open, mock_print):
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_fitz_open.return_value = mock_doc

        with patch("sys.argv", ["pdf_ocr_backend.py", self.test_pdf, "markdown"]):
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(Path, "exists", return_value=True):
                    with self.assertRaises(SystemExit) as cm:
                        pdf_ocr_backend.main()
                    self.assertEqual(cm.exception.code, 1)

        calls = mock_print.call_args_list
        self.assertTrue(
            any("DEEPSEEK_OCR_BASE_URL not set" in str(call) for call in calls)
        )


class TestExitCode3(unittest.TestCase):
    """Unit tests for Exit 3 (NO_OCR_SUPPORT) scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = os.path.join(self.temp_dir, "test.pdf")
        # Create a minimal valid PDF for fitz.open()
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(self.test_pdf)
        doc.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_exit_3_only_when_current_model_configured_no_vision(
        self, mock_get_models, mock_load_config, mock_multimodal
    ):
        """Verify Exit 3 only occurs when current_model configured but no vision support."""
        mock_get_models.return_value = ["text-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"text-model": "current_model"},
            "default": "current_model",
        }
        mock_multimodal.return_value = False

        with self.assertRaises(SystemExit) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(cm.exception.code, 3)

    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_exit_3_error_message_format(
        self, mock_get_models, mock_load_config, mock_multimodal
    ):
        """Verify error message includes model name and configuration instructions."""
        mock_get_models.return_value = ["my-text-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"my-text-model": "current_model"},
            "default": "current_model",
        }
        mock_multimodal.return_value = False

        with patch("sys.stderr") as mock_stderr:
            with self.assertRaises(SystemExit):
                pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

            # Check that error message was printed to stderr
            mock_stderr.write.assert_called()
            error_output = ""
            for call in mock_stderr.write.call_args_list:
                args = call[0]
                if args:
                    error_output += str(args[0])

            self.assertIn("NO_OCR_SUPPORT", error_output)
            self.assertIn("my-text-model", error_output)
            self.assertIn("ocr_routing.json", error_output)
            self.assertIn("loaded models", error_output)


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for full route_ocr_request workflow with model-set routing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = os.path.join(self.temp_dir, "test.pdf")
        # Create a minimal valid PDF for fitz.open()
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(self.test_pdf)
        doc.close()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_integration_kimi_qwen35b_routes_to_deepseek_ocr2(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """Kimi-K2.6 + Qwen3.6-35B-A3B loaded -> routes to deepseek-ocr-2."""
        mock_get_models.return_value = ["kimi-k2.6", "qwen3.6-35b-a3b-nvfp4"]
        mock_load_config.return_value = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-35b-a3b-nvfp4": "deepseek-ocr-2",
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
                "kimi-k2.6": "current_model",
            },
            "default": "current_model",
        }
        mock_process.return_value = "DeepSeek-OCR-2 result"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "DeepSeek-OCR-2 result")
        mock_process.assert_called_once_with(
            self.test_pdf, "markdown", "deepseek-ocr-2", "http://test:8080/v1", [0]
        )

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_integration_kimi_qwen27b_routes_to_glm_ocr(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """Kimi-K2.6 + Qwen3.6-27B loaded -> routes to glm-ocr."""
        mock_get_models.return_value = ["kimi-k2.6", "qwen3.6-27b-nvfp4"]
        mock_load_config.return_value = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-35b-a3b-nvfp4": "deepseek-ocr-2",
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
                "kimi-k2.6": "current_model",
            },
            "default": "current_model",
        }
        mock_process.return_value = "GLM-OCR result"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "GLM-OCR result")
        mock_process.assert_called_once_with(
            self.test_pdf, "markdown", "glm-ocr", "http://test:8080/v1", [0]
        )

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_integration_kimi_only_routes_to_current_model(
        self, mock_get_models, mock_load_config, mock_multimodal, mock_process
    ):
        """Kimi-K2.6 only loaded -> routes to current_model with vision check."""
        mock_get_models.return_value = ["kimi-k2.6"]
        mock_load_config.return_value = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-35b-a3b-nvfp4": "deepseek-ocr-2",
                "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
                "kimi-k2.6": "current_model",
            },
            "default": "current_model",
        }
        mock_multimodal.return_value = True
        mock_process.return_value = "Current model OCR result"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "Current model OCR result")
        mock_process.assert_called_once_with(
            self.test_pdf, "markdown", "kimi-k2.6", "http://test:8080/v1", [0]
        )

    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_integration_no_models_loaded(
        self, mock_get_models, mock_load_config
    ):
        """No models loaded -> raises 'No model currently loaded' exception."""
        mock_get_models.return_value = []
        mock_load_config.return_value = {
            "ocr_routing": {
                "kimi-k2.6,qwen3.6-35b-a3b-nvfp4": "deepseek-ocr-2",
            },
            "default": "current_model",
        }

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertIn("No model currently loaded", str(cm.exception))


class TestBackwardCompatibility(unittest.TestCase):
    """Explicit backward compatibility verification tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = os.path.join(self.temp_dir, "test.pdf")
        # Create a minimal valid PDF for fitz.open()
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(self.test_pdf)
        doc.close()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_single_model_keys_without_commas_work_as_before(self):
        """Old-style config with only single-model keys works unchanged."""
        config = {
            "ocr_routing": {
                "kimi-k2.5": "deepseek-ocr-2",
                "kimi-k2.5-abliterated": "deepseek-ocr-2"
            },
            "default": "current_model"
        }
        # Using new model-set function with old config
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["kimi-k2.5"], config
        )
        self.assertEqual(result, "deepseek-ocr-2")

    def test_existing_config_without_model_set_keys_works(self):
        """Config without any model-set keys (no commas) works without modification."""
        config = {
            "ocr_routing": {
                "some-model": "deepseek-ocr"
            },
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["some-model"], config
        )
        self.assertEqual(result, "deepseek-ocr")

    def test_default_key_behavior_unchanged(self):
        """Default key still falls back correctly when no match found."""
        config = {
            "ocr_routing": {"other-model": "deepseek-ocr"},
            "default": "current_model"
        }
        result = pdf_ocr_backend.get_ocr_method_for_model_set(
            ["unknown-model"], config
        )
        self.assertEqual(result, "current_model")

    @patch("pdf_ocr_backend.requests.get")
    def test_get_current_model_returns_first_loaded_model(self, mock_get):
        """get_current_model() is retained and returns first loaded model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "running": [
                {"model": "first-model"},
                {"model": "second-model"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = pdf_ocr_backend.get_current_model("http://test:8080")

        self.assertEqual(result, "first-model")

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_route_ocr_request_signature_unchanged(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """route_ocr_request() signature accepts same positional args as before."""
        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.return_value = "Success"

        # Call with same signature as before: pdf_path, output_format
        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "Success")

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_exit_code_0_unchanged_success(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """Exit code 0 still indicates successful OCR completion."""
        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.return_value = "OCR completed successfully"

        result = pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(result, "OCR completed successfully")

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_exit_code_1_unchanged_general_error(
        self, mock_get_models, mock_load_config, mock_process
    ):
        """Exit code 1 still indicates general error (e.g., OCR failure)."""
        mock_get_models.return_value = ["test-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"test-model": "deepseek-ocr"},
            "default": "current_model",
        }
        mock_process.side_effect = Exception("OCR service failed")

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertIn("OCR service failed", str(cm.exception))

    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_exit_code_3_unchanged_no_ocr_support(
        self, mock_get_models, mock_load_config, mock_multimodal
    ):
        """Exit code 3 still indicates NO_OCR_SUPPORT with updated message."""
        mock_get_models.return_value = ["text-only-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"text-only-model": "current_model"},
            "default": "current_model",
        }
        mock_multimodal.return_value = False

        with self.assertRaises(SystemExit) as cm:
            pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

        self.assertEqual(cm.exception.code, 3)

    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"})
    def test_exit_code_3_error_message_mentions_model_sets(
        self, mock_get_models, mock_load_config, mock_multimodal
    ):
        """Exit code 3 error message mentions loaded model sets."""
        mock_get_models.return_value = ["text-only-model"]
        mock_load_config.return_value = {
            "ocr_routing": {"text-only-model": "current_model"},
            "default": "current_model",
        }
        mock_multimodal.return_value = False

        with patch("sys.stderr") as mock_stderr:
            with self.assertRaises(SystemExit):
                pdf_ocr_backend.route_ocr_request(self.test_pdf, "markdown")

            error_output = ""
            for call in mock_stderr.write.call_args_list:
                args = call[0]
                if args:
                    error_output += str(args[0])

            self.assertIn("NO_OCR_SUPPORT", error_output)
            self.assertIn("loaded models", error_output)
            self.assertIn("ocr_routing.json", error_output)


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
    @patch("pdf_ocr_backend.fitz.open")
    def test_download_pdf_success(self, mock_fitz_open, mock_get):
        """Test successful PDF download."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "1024"}
        mock_response.iter_content.return_value = [b"fake pdf content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_fitz_open.return_value = mock_doc

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

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/missing.pdf")

        self.assertIn("Failed to download", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    def test_download_pdf_connection_error(self, mock_get):
        """Test download failure on connection error."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/test.pdf")

        self.assertIn("Failed to download", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    def test_download_pdf_content_length_exceeded(self, mock_get):
        """Test download rejected when Content-Length exceeds limit."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": str(100 * 1024 * 1024)}  # 100 MB
        mock_response.iter_content.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/huge.pdf")

        self.assertIn("too large", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.fitz.open")
    def test_download_pdf_streaming_size_exceeded(self, mock_fitz_open, mock_get):
        """Test download rejected when streaming exceeds limit."""
        mock_response = MagicMock()
        mock_response.headers = {}
        # Simulate chunks that exceed 50MB
        mock_response.iter_content.return_value = [b"x" * (1024 * 1024)] * 60
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/huge.pdf")

        self.assertIn("exceeds maximum", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.fitz.open")
    def test_download_pdf_invalid_content(self, mock_fitz_open, mock_get):
        """Test download rejected when content is not a valid PDF."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.iter_content.return_value = [b"not a pdf"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_fitz_open.side_effect = Exception("cannot open document")

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/notpdf.txt")

        self.assertIn("not a valid PDF", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.fitz.open")
    def test_download_pdf_empty_file(self, mock_fitz_open, mock_get):
        """Test download rejected when file is empty."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "0"}
        mock_response.iter_content.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/empty.pdf")

        self.assertIn("empty", str(cm.exception))

    @patch("pdf_ocr_backend.requests.get")
    @patch("pdf_ocr_backend.fitz.open")
    def test_download_pdf_no_pages(self, mock_fitz_open, mock_get):
        """Test download rejected when PDF has no pages."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "1024"}
        mock_response.iter_content.return_value = [b"fake pdf content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_fitz_open.return_value = mock_doc

        with self.assertRaises(Exception) as cm:
            pdf_ocr_backend.download_pdf("https://example.com/empty.pdf")

        self.assertIn("no pages", str(cm.exception).lower())

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

        with patch("sys.argv", ["pdf_ocr_backend.py", "https://example.com/test.pdf", "markdown"]):
            with patch("builtins.print") as mock_print:
                with patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080/v1"}):
                    pdf_ocr_backend.main()

        mock_download.assert_called_once_with("https://example.com/test.pdf")
        mock_route.assert_called_once_with(
            temp_path,
            "markdown",
            None,
            base_url=None,
            routing_config_path=None,
        )
        calls = mock_print.call_args_list
        self.assertTrue(any("OCR result from URL" in str(call) for call in calls))

        # Cleanup if file still exists
        pdf_ocr_backend.cleanup_temp_file(temp_path)

    @patch("pdf_ocr_backend.download_pdf")
    def test_main_with_url_download_failure(self, mock_download):
        """Test main() exits with code 1 when URL download fails."""
        mock_download.side_effect = Exception("Download failed")

        with patch("sys.argv", ["pdf_ocr_backend.py", "https://example.com/bad.pdf", "markdown"]):
            with patch("builtins.print") as mock_print:
                with self.assertRaises(SystemExit) as cm:
                    pdf_ocr_backend.main()
                self.assertEqual(cm.exception.code, 1)

        calls = mock_print.call_args_list
        self.assertTrue(any("Download failed" in str(call) for call in calls))


class TestBaseUrlAndRoutingConfigOverrides(unittest.TestCase):
    """Explicit base-url / routing-config overrides (used by the MCP adapter)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf = os.path.join(self.temp_dir, "test.pdf")
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(self.test_pdf)
        doc.close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://env:8080"})
    def test_explicit_base_url_overrides_env(
        self, mock_get_models, mock_load_config, mock_multimodal, mock_process
    ):
        """An explicit base_url wins over DEEPSEEK_OCR_BASE_URL."""
        mock_get_models.return_value = ["vision-model"]
        mock_load_config.return_value = {"ocr_routing": {}, "default": "current_model"}
        mock_multimodal.return_value = True
        mock_process.return_value = "ok"

        result = pdf_ocr_backend.route_ocr_request(
            self.test_pdf, "markdown", base_url="http://explicit:8080"
        )

        self.assertEqual(result, "ok")
        mock_get_models.assert_called_once_with("http://explicit:8080")
        mock_process.assert_called_once_with(
            self.test_pdf, "markdown", "vision-model", "http://explicit:8080", [0]
        )

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_base_url_works_without_env(
        self, mock_get_models, mock_load_config, mock_multimodal, mock_process
    ):
        """An explicit base_url works even when the env var is unset."""
        mock_get_models.return_value = ["vision-model"]
        mock_load_config.return_value = {"ocr_routing": {}, "default": "current_model"}
        mock_multimodal.return_value = True
        mock_process.return_value = "ok"

        result = pdf_ocr_backend.route_ocr_request(
            self.test_pdf, "markdown", base_url="http://explicit:8080"
        )

        self.assertEqual(result, "ok")

    @patch("pdf_ocr_backend.process_with_model")
    @patch("pdf_ocr_backend.check_multimodal_support")
    @patch("pdf_ocr_backend.load_ocr_routing_config")
    @patch("pdf_ocr_backend.get_all_models")
    @patch.dict(os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://test:8080"})
    def test_routing_config_path_forwarded(
        self, mock_get_models, mock_load_config, mock_multimodal, mock_process
    ):
        """An explicit routing_config_path is forwarded to the loader."""
        mock_get_models.return_value = ["vision-model"]
        mock_load_config.return_value = {"ocr_routing": {}, "default": "current_model"}
        mock_multimodal.return_value = True
        mock_process.return_value = "ok"

        pdf_ocr_backend.route_ocr_request(
            self.test_pdf, "markdown", routing_config_path="/tmp/alt-routing.json"
        )

        mock_load_config.assert_called_once_with(Path("/tmp/alt-routing.json"))

    def test_resolve_routing_config_path_precedence(self):
        """Explicit arg > PDF_OCR_ROUTING_CONFIG env > tool-dir default."""
        explicit = Path("/tmp/explicit.json")
        self.assertEqual(
            pdf_ocr_backend.resolve_routing_config_path(explicit), explicit
        )

        with patch.dict(os.environ, {"PDF_OCR_ROUTING_CONFIG": "/tmp/env.json"}):
            self.assertEqual(
                pdf_ocr_backend.resolve_routing_config_path(), Path("/tmp/env.json")
            )

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                pdf_ocr_backend.resolve_routing_config_path(),
                Path(pdf_ocr_backend.__file__).parent / "ocr_routing.json",
            )

    def test_load_ocr_routing_config_honors_env(self):
        """load_ocr_routing_config() uses PDF_OCR_ROUTING_CONFIG when unset."""
        config_content = json.dumps(
            {"ocr_routing": {"m": "glm-ocr"}, "default": "current_model"}
        )

        with patch.dict(os.environ, {"PDF_OCR_ROUTING_CONFIG": "/tmp/env-routing.json"}):
            with patch("builtins.open", mock_open(read_data=config_content)) as m:
                with patch.object(Path, "exists", return_value=True):
                    result = pdf_ocr_backend.load_ocr_routing_config()

        self.assertEqual(result["ocr_routing"]["m"], "glm-ocr")
        m.assert_called_once_with(Path("/tmp/env-routing.json"), "r")


if __name__ == "__main__":
    unittest.main()
