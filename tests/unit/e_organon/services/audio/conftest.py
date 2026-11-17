"""tests/unit/e_organon/services/audio: the vault the transcribe tool reads — a recording that is only a name to the
fakes, its roots, and the engine of the tests' own scripted per test."""

from typing import TYPE_CHECKING

import pytest

from e_organon.engines.artifact import ArtifactEngine
from e_organon.services.files import AllowedFiles

if TYPE_CHECKING:
    from pathlib import Path


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
