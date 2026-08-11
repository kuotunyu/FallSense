from __future__ import annotations

import re
from pathlib import Path

import yaml


def test_release_metadata_is_present_and_consistent() -> None:
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "MIT"
    assert citation["references"][0]["doi"] == "10.3390/s19091988"
    assert len(citation["references"][0]["authors"]) == 6

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in pyproject
    assert 'license = { text = "MIT" }' in pyproject
    assert "MIT License" in license_text


def test_release_documents_and_architecture_exist() -> None:
    required = (
        "MODEL_CARD.md",
        "DATA_CARD.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/architecture.md",
    )
    for name in required:
        path = Path(name)
        assert path.is_file() and path.stat().st_size > 0, name


def test_model_card_uses_registered_headline_metrics() -> None:
    model_card = Path("MODEL_CARD.md").read_text(encoding="utf-8")
    assert "EXP-GBM-BIN-LOSO-001" in model_card
    assert "per-held-out-subject mean±std" in model_card
    for metric in ("macro-F1", "fall recall", "precision", "AUPRC", "false-positive rate"):
        assert metric in model_card
    assert "Vision/fusion performance 沒有訓練" in model_card


def test_release_document_local_links_resolve() -> None:
    documents = (
        Path("README.md"),
        Path("MODEL_CARD.md"),
        Path("DATA_CARD.md"),
        Path("docs/architecture.md"),
    )
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", content):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            assert (document.parent / relative).exists(), f"{document}: {target}"
