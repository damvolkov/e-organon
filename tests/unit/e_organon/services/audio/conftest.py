"""tests/unit/e_organon/services/audio: the vault the transcribe tool reads — a recording that is only a name to the
fakes, its roots, and `build`, the tool over the fakes it is handed: a window of twenty minutes, two of overlap, two
retries, unless a test says otherwise."""

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from e_organon.engines.artifact import ArtifactEngine
from e_organon.ports.media import Frame, MediaInfo
from e_organon.ports.ocr import OcrWord
from e_organon.processors.speakers import SpeakersProcessor
from e_organon.services.audio.transcribe import TranscribeAudioService
from e_organon.services.files import AllowedFiles
from support.media import FakeMedia
from support.ocr import FakeOcr
from support.stage import FakeStage, ring, strip

if TYPE_CHECKING:
    from pathlib import Path

    from support.transcribe import Build, FakeTranscribe


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "mitin.mp4").write_bytes(b"not a container: the media reader here is a fake")
    return tmp_path


@pytest.fixture
def files(vault: Path) -> AllowedFiles:
    return AllowedFiles((vault,))


@pytest.fixture
def artifacts(vault: Path) -> ArtifactEngine:
    return ArtifactEngine(vault / "outbox")


@pytest.fixture
def build(files: AllowedFiles, artifacts: ArtifactEngine) -> Build:
    def make(
        engine: FakeTranscribe,
        media: FakeMedia,
        *,
        window: timedelta = timedelta(minutes=20),
        overlap: timedelta = timedelta(minutes=2),
        retries: int = 2,
        speakers: SpeakersProcessor | None = None,
        frames: tuple[timedelta, int, int] | None = None,
    ) -> TranscribeAudioService:
        return TranscribeAudioService(
            engine,
            media,
            files,
            artifacts,
            limit=timedelta(minutes=90),
            window=window,
            overlap=overlap,
            retries=retries,
            speakers=speakers,
            frames=frames,
        )

    return make


@pytest.fixture
def stage() -> FakeStage:
    """One tile speaking at 1, 2 and 3 s, its strip looking the same each time."""
    return FakeStage({at: (ring((0, 0, 64, 36), 1),) for at in (1.0, 2.0, 3.0)})


@pytest.fixture
def ocr() -> FakeOcr:
    """The strip of that tile reads as a Teams label: the name, then the organisation in brackets."""
    return FakeOcr(readings={strip(1): (OcrWord(text="Ana García (Allianz)", box=(0, 0, 8, 8)),)})


@pytest.fixture
def speakers(stage: FakeStage, ocr: FakeOcr) -> SpeakersProcessor:
    return SpeakersProcessor(stage, ocr, width=64, height=36, alike=10.0, votes=3, share=0.6)


@pytest.fixture
def filmed() -> FakeMedia:
    """A ten-second recording with a picture each at 1, 2 and 3 s."""
    return FakeMedia(
        MediaInfo(duration=10.0, has_audio=True, has_video=True),
        pictures=tuple(Frame(at=at, pixels=b"") for at in (1.0, 2.0, 3.0)),
    )
