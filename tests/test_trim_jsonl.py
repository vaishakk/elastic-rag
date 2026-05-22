from __future__ import annotations

from pathlib import Path

import pytest

from trim_jsonl import trim_jsonl


def test_trim_jsonl_writes_inclusive_doc_range(tmp_path: Path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "test-docs.jsonl"
    src.write_text(
        '{"id": 1, "text": "one"}\n'
        '\n'
        '{"id": 2, "text": "two"}\n'
        '{"id": 3, "text": "three"}\n'
        '{"id": 4, "text": "four"}\n',
        encoding="utf-8",
    )

    written = trim_jsonl(src, dst, start_doc=2, end_doc=3)

    assert written == 2
    assert dst.read_text(encoding="utf-8") == (
        '{"id": 2, "text": "two"}\n'
        '{"id": 3, "text": "three"}\n'
    )


def test_trim_jsonl_rejects_invalid_range(tmp_path: Path):
    src = tmp_path / "input.jsonl"
    dst = tmp_path / "test-docs.jsonl"
    src.write_text('{"id": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="--end-doc must be >= --start-doc"):
        trim_jsonl(src, dst, start_doc=3, end_doc=2)
