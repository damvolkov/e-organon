"""tests/unit/e_organon/api/lifespan/services: crypto written to the graph, no resource to close; the transcribe tool built
on the settings, naming speakers only with an OCR."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from e_organon.api.lifespan import services
from e_organon.api.state import AppState
from e_organon.engines.artifact import ArtifactEngine
from e_organon.infra.state import State
from e_organon.models.audio import TranscribeIn
from e_organon.ops.crypto import Crypto
from e_organon.ports.media import Frame, MediaInfo
from e_organon.services.files import AllowedFiles
from e_organon.services.flow import ToolFlow
from support.media import FakeMedia
from support.ocr import FakeOcr, said
from support.stage import FakeStage, ring, strip
from support.transcribe import FakeTranscribe, seg, transcript

if TYPE_CHECKING:
    from pathlib import Path

    from litestar import Litestar
    from litestar.testing import TestClient

TEKTON = ("eng", "start", "status", "tasks", "cancel", "catalog", "find", "read", "update", "peers", "send")


async def test_services_write_crypto_to_the_graph() -> None:
    """Crypto is the one engine the services concern writes; it owns no resource to close."""
    app = SimpleNamespace(state=AppState({"system": State()}))
    async with services.lifespan(cast("Litestar", app)):
        assert isinstance(app.state.system.crypto, Crypto)


async def test_services_crypto_hashes_what_it_verifies() -> None:
    app = SimpleNamespace(state=AppState({"system": State()}))
    async with services.lifespan(cast("Litestar", app)):
        crypto = app.state.system.crypto
        hashed = await crypto.hash_password("correct-horse-1")
        assert await crypto.verify_password("correct-horse-1", hashed)


async def test_services_tekton_tools_hang_at_their_names(client: TestClient) -> None:
    """Each tool under `services.tekton.<tool>`; `update`, a method of the state, is read by key."""
    tekton = client.app.state.system.services.tekton
    assert (sorted(tekton), all(isinstance(tekton[tool], ToolFlow) for tool in TEKTON)) == (sorted(TEKTON), True)


@pytest.mark.parametrize(("with_ocr", "named"), [(True, {"S01": "Ana García"}), (False, {})], ids=["ocr", "no-ocr"])
async def test_services_transcriber_names_speakers_only_with_an_ocr(
    tmp_path: Path, with_ocr: bool, named: dict[str, str]
) -> None:
    (tmp_path / "mitin.mp4").write_bytes(b"not a container: the media reader here is a fake")
    place = (0, 0, 64, 36)
    stage = FakeStage({at: (ring(place, 1),) for at in (1.0, 2.0, 3.0)})
    ocr = FakeOcr(readings={strip(1): said("Ana García (Allianz)")}) if with_ocr else None
    media = FakeMedia(
        MediaInfo(duration=10.0, has_audio=True, has_video=True),
        pictures=tuple(Frame(at=at, pixels=b"") for at in (1.0, 2.0, 3.0)),
    )
    tool = services.transcriber(
        FakeTranscribe(transcript((seg(0.5, 3.5, "S01", "Hola"),))),
        media,
        AllowedFiles((tmp_path,)),
        ArtifactEngine(tmp_path / "outbox"),
        stage=stage,
        ocr=ocr,
    )
    out = await tool.run(TranscribeIn(path=str(tmp_path / "mitin.mp4")))
    assert (out.speakers, out.windows) == (named, 1)
