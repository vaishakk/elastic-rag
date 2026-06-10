from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    title: str
    text: str
    url: Path | None = None

@dataclass
class SearchResults:
    docs: List[Document]

@dataclass
class Query:
    query: str

@dataclass
class InferenceResults:
    answer: str
    refs: List[Path] | None = None

@dataclass
class InferQuery:
    query: str
    context: List[Document]
