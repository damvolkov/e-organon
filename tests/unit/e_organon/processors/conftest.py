"""tests/unit/e_organon/processors: the speakers processor over a stage and an OCR of the tests' own — it asks three votes
and six in ten, and calls two strips alike under ten grey levels of mean difference."""

import pytest

from e_organon.processors.speakers import SpeakersProcessor
from support.ocr import FakeOcr
from support.stage import FakeStage


@pytest.fixture
def stage() -> FakeStage:
    return FakeStage()


@pytest.fixture
def ocr() -> FakeOcr:
    return FakeOcr()


@pytest.fixture
def speakers(stage: FakeStage, ocr: FakeOcr) -> SpeakersProcessor:
    return SpeakersProcessor(stage, ocr, width=64, height=36, alike=10.0, votes=3, share=0.6)
