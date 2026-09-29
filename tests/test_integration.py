#!/usr/bin/env python3
"""
Test Case 2: Multi-Page Document
Tests multi-page PDF processing functionality.
"""

import subprocess
import sys
from pathlib import Path


def test_multipage_pdf():
    """Test multi-page PDF processing."""
    pdf_path = Path(__file__).parent / "test_data" / "multipage.pdf"

    if not pdf_path.exists():
        print("❌ Test PDF not found")
        return False

    print("Testing multi-page PDF processing...")

    # Test with markdown output
    print("\n1. Testing markdown output format...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(Path(__file__).parent.parent),
            "python3",
            "tool/pdf_ocr_backend.py",
            str(pdf_path),
            "markdown",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return False

    output = result.stdout

    # Verify all pages are processed
    expected_pages = 10
    page_markers = output.count("--- Page ")
    if page_markers != expected_pages:
        print(f"❌ Expected {expected_pages} page markers, found {page_markers}")
        return False
    print(f"✅ All {expected_pages} pages processed")

    # Verify page break markers are present
    if "--- Page 1 ---" not in output:
        print("❌ Page 1 marker missing")
        return False
    if "--- Page 10 ---" not in output:
        print("❌ Page 10 marker missing")
        return False
    print("✅ Page break markers present")

    # Verify page concatenation
    pages = output.split("\n\n--- Page ")
    if len(pages) != expected_pages:
        print(f"❌ Expected {expected_pages} pages in output, found {len(pages)}")
        return False
    print("✅ Page concatenation correct")

    # Verify formatting consistency
    for i in range(1, expected_pages + 1):
        if f"--- Page {i} ---" not in output:
            print(f"❌ Page {i} marker missing")
            return False
    print("✅ Formatting consistent across pages")

    # Verify content is present
    if "PrintNode Multi Page Test Start" not in output:
        print("❌ Start content missing")
        return False
    if "PrintNode" not in output:
        print("❌ End content missing")
        return False
    print("✅ Content extracted correctly")

    # Test with plain text output
    print("\n2. Testing plain text output format...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(Path(__file__).parent.parent),
            "python3",
            "tool/pdf_ocr_backend.py",
            str(pdf_path),
            "text",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return False

    text_output = result.stdout

    # Verify same page count
    text_page_markers = text_output.count("--- Page ")
    if text_page_markers != expected_pages:
        print(
            f"❌ Expected {expected_pages} page markers in text mode, found {text_page_markers}"
        )
        return False
    print(f"✅ Text output: All {expected_pages} pages processed")

    # Compare outputs (should be similar structure)
    if text_page_markers != page_markers:
        print("❌ Page count mismatch between markdown and text output")
        return False
    print("✅ Output consistency between formats")

    print("\n✅ All multi-page PDF tests passed!")
    return True


if __name__ == "__main__":
    success = test_multipage_pdf()
    sys.exit(0 if success else 1)
