"""Multi-format file reader. Returns (raw_text, file_type)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import defusedxml.ElementTree as SafeET
import pandas as pd
import structlog

from src.tools.pdf_extractor import extract_text as extract_pdf_text

logger = structlog.get_logger(__name__)


def parse_file(file_path: str) -> tuple[str, str]:
    """Detect file format by extension and return (text, file_type)."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    _read_text = lambda p: Path(p).read_text(encoding="utf-8", errors="replace")

    dispatch = {
        ".txt": ("txt", _read_text),
        ".json": ("json", lambda p: json.dumps(json.load(open(p, encoding="utf-8")), indent=2)),
        ".csv": ("csv", lambda p: pd.read_csv(p, dtype=str).to_string(index=False)),
        ".xml": ("xml", _read_xml),
        ".pdf": ("pdf", _read_pdf),
        ".eml": ("eml", _read_text),
        ".email": ("eml", _read_text),
    }

    if ext not in dispatch:
        raise ValueError(f"Unsupported file format '{ext}' for file: {file_path}")

    file_type, reader = dispatch[ext]
    text = reader(file_path)
    logger.debug("file_parsed", file=file_path, file_type=file_type, chars=len(text))
    return text, file_type


def _read_xml(file_path: str) -> str:
    tree = SafeET.parse(file_path)
    root = tree.getroot()
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode")


def _read_pdf(file_path: str) -> str:
    text, warnings = extract_pdf_text(file_path)
    if warnings:
        warning_block = "\n\n[PARSER WARNINGS]\n" + "\n".join(f"- {w}" for w in warnings)
        text = text + warning_block if text else warning_block
    return text
