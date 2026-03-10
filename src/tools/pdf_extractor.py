"""PDF text extraction via pdfplumber."""

import pdfplumber
import structlog

logger = structlog.get_logger(__name__)


def extract_text(file_path: str) -> tuple[str, list[str]]:
    """Pull text from all pages. Returns (text, warnings)."""
    warnings: list[str] = []
    pages_text: list[str] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                warnings.append("PDF has no pages")
                return "", warnings

            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    warnings.append(f"Page {i} appears blank or image only (scanned PDF?)")
                pages_text.append(page_text)

        full_text = "\n".join(pages_text).strip()
        if not full_text:
            warnings.append("No text extracted — document may be a scanned image.")

        return full_text, warnings

    except Exception as exc:
        msg = f"Failed to extract from PDF '{file_path}': {exc}"
        logger.warning("pdf_extract_failed", file=file_path, error=str(exc))
        return "", [msg]
