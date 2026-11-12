"""The two datasets the project knows: a small synthetic fixture and the real corpus."""

from dataclasses import dataclass, replace
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
    # Best reranker score below which the answer is refused. Scores depend on the corpus, so each
    # dataset has its own. `REFUSAL_THRESHOLD` in the environment overrides it.
    refusal_threshold: float


FIXTURE = Dataset(
    name="fixture",
    corpus_path=ROOT / "tests" / "fixtures" / "tk_fixture.txt",
    document="tk-fixture",
    golden_path=ROOT / "tests" / "golden_fixture.jsonl",
    report_stem="results",
    refusal_threshold=0.003,  # calibrated on the fixture golden (a toy corpus)
)


# Calibrated on the DEV half of the golden set only (src/tkrag/eval/experiments.py: split_dev_test,
# seed 13): the midpoint between the highest no_answer score and the lowest answerable score
# there. The TEST half was not used. DESIGN.md D7.
FULL_REFUSAL_THRESHOLD = 0.0069


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
        refusal_threshold=FULL_REFUSAL_THRESHOLD,
    )


def get_dataset(
    name: str,
    settings: Settings,
    golden: Path | None = None,
    stem: str | None = None,
) -> Dataset:
    """`golden` and `stem` point the same corpus at another question set and report name."""
    if name == "fixture":
        dataset = FIXTURE
    elif name == "full":
        dataset = full_dataset(settings)
    else:
        raise ValueError(f"unknown dataset {name!r}: use 'fixture' or 'full'")
    if golden is not None:
        dataset = replace(dataset, golden_path=golden)
    if stem is not None:
        dataset = replace(dataset, report_stem=stem)
    return dataset
