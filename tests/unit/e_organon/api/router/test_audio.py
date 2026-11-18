"""tests/unit/e_organon/api/router/audio: audio/transcribe over REST — no route until the sidecar is configured, then a
recording heard in windows answers its segments and where they end."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from litestar.testing import TestClient

    from support.transcribe import FakeTranscribe


def test_transcribe_is_no_route_until_the_sidecar_is_on(
    vault: Path, client: TestClient, ana_key: dict[str, str]
) -> None:
    response = client.post("/audio/transcribe", json={"path": str(vault / "rec.mp4")}, headers=ana_key)
    assert response.status_code == 404


def test_transcribe_answers_the_segments_the_windows_and_where_speech_ends(
    vault: Path, moss_on: FakeTranscribe, client: TestClient, ana_key: dict[str, str]
) -> None:
    (vault / "rec.mp4").write_bytes(b"not a container: the media reader here is a fake")
    response = client.post("/audio/transcribe", json={"path": str(vault / "rec.mp4")}, headers=ana_key)
    out = response.json()
    assert response.status_code == 200
    assert (out["windows"], out["covered"], out["speakers"]) == (1, 2.5, {})
    assert [(each["speaker"], each["text"], each["name"]) for each in out["segments"]] == [("S01", "Hola", "")]
