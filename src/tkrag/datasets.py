"""The two datasets the project knows: a small synthetic fixture and the real corpus."""

from dataclasses import dataclass
from pathlib import Path

from tkrag.config import Settings

ROOT = Path(__file__).resolve().parents[2]
_CORPUS_SUFFIXES = {".txt", ".html", ".htm"}


@dataclass(frozen=True)
class Dataset:
    name: str  # "fixture" | "full"
    corpus_path: Path
    document: str  # value of documents.name in the DB
    golden_path: Path
    report_stem: str  # reports/<stem>.md and .json


FIXTURE = Dataset(
    name="fixture",
    corpus_path=ROOT / "tests" / "fixtures" / "tk_fixture.txt",
    document="tk-fixture",
    golden_path=ROOT / "tests" / "golden_fixture.jsonl",
    report_stem="results",
)


class CorpusNotFoundError(FileNotFoundError):
    pass


def full_dataset(settings: Settings) -> Dataset:
    raw_dir = ROOT / settings.raw_data_dir
    candidates = (
        sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in _CORPUS_SUFFIXES)
        if raw_dir.is_dir()
        else []
    )
    if not candidates:
        raise CorpusNotFoundError(
            f"no corpus file (.txt/.html) in {raw_dir}. Put the Labor Code there "
            "together with SOURCE.md, or use --dataset fixture."
        )
    return Dataset(
        name="full",
        corpus_path=candidates[0],
        document="tk-rf",
        golden_path=ROOT / settings.golden_path,
        report_stem="results_full",
    )


def get_dataset(name: str, settings: Settings) -> Dataset:
    if name == "fixture":
        return FIXTURE
    if name == "full":
        return full_dataset(settings)
    raise ValueError(f"unknown dataset {name!r}: use 'fixture' or 'full'")
