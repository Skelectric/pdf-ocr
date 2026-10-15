"""Shared helpers for the test suite.

The tests are ``unittest.TestCase`` classes, so pytest fixtures cannot be
injected into them. These are plain helpers and a base class instead, which
keeps the PDF fixture, the route fixture and the download-response mock in one
place.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pymupdf


def make_pdf(directory, pages=1, name="test.pdf"):
    """Write a real PDF with ``pages`` blank pages and return its path."""
    path = os.path.join(directory, name)
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=612, height=792)
    doc.save(path)
    doc.close()
    return path


def write_route(directory, route, name="ocr_routing.json"):
    """Write a route file and return its path."""
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        json.dump(route, f)
    return path


def download_response(headers=None, chunks=None):
    """A stand-in for the response of ``requests.get`` in ``download_pdf``."""
    response = MagicMock()
    response.headers = headers or {}
    response.iter_content.return_value = chunks or []
    response.raise_for_status = MagicMock()
    return response


class TempDirTestCase(unittest.TestCase):
    """A TestCase with a temp directory that is removed on teardown."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)