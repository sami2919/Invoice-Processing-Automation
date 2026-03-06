"""Tests for batch file deduplication (Session 5).

Invoices like INV-1011 exist as both .txt and .pdf. Batch processing
should deduplicate by stem, preferring non-PDF formats.
"""

import tempfile
from pathlib import Path

from app import collect_batch_files


class TestCollectBatchFiles:
    """Test stem-based deduplication in batch file collection."""

    def _create_files(self, tmp: Path, names: list[str]) -> None:
        for name in names:
            (tmp / name).write_text(f"contents of {name}")

    def test_no_duplicates_when_only_one_format(self, tmp_path: Path):
        self._create_files(tmp_path, ["inv_1001.txt", "inv_1002.json"])
        result = collect_batch_files(tmp_path)
        assert len(result) == 2

    def test_dedup_prefers_txt_over_pdf(self, tmp_path: Path):
        self._create_files(tmp_path, ["inv_1011.txt", "inv_1011.pdf"])
        result = collect_batch_files(tmp_path)
        assert len(result) == 1
        assert result[0].endswith(".txt")

    def test_dedup_prefers_json_over_pdf(self, tmp_path: Path):
        self._create_files(tmp_path, ["inv_1012.json", "inv_1012.pdf"])
        result = collect_batch_files(tmp_path)
        assert len(result) == 1
        assert result[0].endswith(".json")

    def test_dedup_prefers_csv_over_pdf(self, tmp_path: Path):
        self._create_files(tmp_path, ["inv_1013.csv", "inv_1013.pdf"])
        result = collect_batch_files(tmp_path)
        assert len(result) == 1
        assert result[0].endswith(".csv")

    def test_keeps_pdf_when_no_text_alternative(self, tmp_path: Path):
        self._create_files(tmp_path, ["inv_1014.pdf"])
        result = collect_batch_files(tmp_path)
        assert len(result) == 1
        assert result[0].endswith(".pdf")

    def test_multiple_stems_with_mixed_duplicates(self, tmp_path: Path):
        self._create_files(tmp_path, [
            "inv_1011.txt", "inv_1011.pdf",
            "inv_1012.json", "inv_1012.pdf",
            "inv_1001.txt",
            "inv_1003.xml",
        ])
        result = collect_batch_files(tmp_path)
        stems = [Path(f).stem for f in result]
        assert len(result) == 4
        assert len(set(stems)) == 4  # all unique stems

    def test_returns_sorted(self, tmp_path: Path):
        self._create_files(tmp_path, ["c.txt", "a.txt", "b.json"])
        result = collect_batch_files(tmp_path)
        assert result == sorted(result)

    def test_empty_directory(self, tmp_path: Path):
        result = collect_batch_files(tmp_path)
        assert result == []
