"""tests/unit/e_organon/providers/drivers/speech/moss: the sidecar asked the OpenAI way — a multipart body streamed from
the caller's generator with the engine's own fields, the `verbose_json` answer typed segment by segment, and every shape
outside the contract refused as the server's own words."""

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import HttpUrl

from e_organon.infra.errors import TranscribeError
from e_organon.providers.drivers.speech.moss import CONTENT_TYPE, HOTWORDS, MULTIPART, PROMPT_BASE, MossDriver
from support.http import FakeHTTP

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

ANSWER: dict[str, object] = {
    "text": "[0.50][S01]Hola[1.20] [2.00][S02]Adiós[3.50]",
    "language": "es",
    "segments": [
        {"id": 0, "seek": 0, "start": 0.5, "end": 1.2, "text": "[S01] Hola"},
        {"id": 1, "seek": 0, "start": 2.0, "end": 3.5, "text": "[S02] Adiós"},
    ],
}
URL = HttpUrl("http://127.0.0.1:1/v1")


async def audio() -> AsyncIterator[bytes]:
    yield b"RIFF"
    yield b"fake-wav"


@pytest.fixture
def http() -> FakeHTTP:
    return FakeHTTP({"/audio/transcriptions": ANSWER, "/models": {"data": []}})


@pytest.fixture
def driver(http: FakeHTTP) -> MossDriver:
    return MossDriver(http, url=URL, model="moss-test", deadline=timedelta(minutes=2), max_new_tokens=65_536)


async def test_moss_streams_the_wav_inside_one_multipart_request(driver: MossDriver, http: FakeHTTP) -> None:
    transcript = await driver.transcribe(audio(), name="mitin", language="es", hotwords=("Kanban", "e-organon"))
    ((url, body, within, headers),) = http.uploads
    assert transcript.language == "es"
    assert (url, within) == ("http://127.0.0.1:1/v1/audio/transcriptions", timedelta(minutes=2))
    ### the header is what makes the streamed body a form: its boundary must be the one the body separates on.
    declared = body.split(b"\r\n", 1)[0].removeprefix(b"--").decode()
    assert headers == {CONTENT_TYPE: MULTIPART.format(declared)}
    assert body.endswith(f"--{declared}--\r\n".encode())
    assert (b'form-data; name="file"; filename="mitin.wav"' in body, b"RIFFfake-wav" in body) == (True, True)
    for field in (b'name="model"', b'name="response_format"', b"verbose_json", b'name="max_new_tokens"', b"65536"):
        assert field in body
    assert (b"Kanban, e-organon" in body, b'name="language"\r\n\r\nes' in body) == (True, True)
    ### the hotwords append the card's hint to the trained prompt; the hint alone would replace it.
    assert (PROMPT_BASE + HOTWORDS + "Kanban, e-organon").encode() in body


async def test_moss_types_the_verbose_answer(driver: MossDriver, http: FakeHTTP) -> None:
    found = await driver.transcribe(audio(), name="mitin")
    assert (found.language, len(found.segments)) == ("es", 2)
    first, second = found.segments
    assert (first.start, first.speaker, first.text) == (0.5, "S01", "Hola")
    assert (second.end, second.speaker, second.text) == (3.5, "S02", "Adiós")
    assert found.seconds >= 0


async def test_moss_reads_the_speaker_from_its_field_when_the_server_carries_one() -> None:
    server = FakeHTTP(
        {
            "/audio/transcriptions": {
                "text": "Hola Adiós",
                "segments": [
                    {"start": 0.5, "end": 1.2, "text": "Hola", "speaker": "S01"},
                    {"start": 2.0, "end": 3.5, "text": "Adiós", "speaker_id": 1},
                ],
            }
        }
    )
    driver = MossDriver(server, url=URL, model="moss-test", deadline=timedelta(seconds=5), max_new_tokens=1024)
    found = await driver.transcribe(audio(), name="mitin")
    assert [(each.speaker, each.text) for each in found.segments] == [("S01", "Hola"), ("S02", "Adiós")]


async def test_moss_one_voice_heard_names_none() -> None:
    server = FakeHTTP(
        {
            "/audio/transcriptions": {
                "text": "solo",
                "language": "auto",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}],
            }
        }
    )
    driver = MossDriver(server, url=URL, model="moss-test", deadline=timedelta(seconds=5), max_new_tokens=1024)
    found = await driver.transcribe(audio(), name="mitin")
    assert (found.language, [(each.speaker, each.text) for each in found.segments]) == ("", [("", "hola")])


async def test_moss_sends_no_language_and_no_prompt_when_asked_of_none(driver: MossDriver, http: FakeHTTP) -> None:
    await driver.transcribe(audio(), name="mitin")
    ((_, body, _, _),) = http.uploads
    assert (b'name="language"' in body, b'name="prompt"' in body) == (False, False)


async def test_moss_refuses_an_answer_outside_the_contract() -> None:
    broken = MossDriver(
        FakeHTTP({"/audio/transcriptions": {"text": "sin segmentos"}}),
        url=URL,
        model="moss-test",
        deadline=timedelta(seconds=5),
        max_new_tokens=1024,
    )
    with pytest.raises(TranscribeError, match="verbose_json"):
        await broken.transcribe(audio(), name="mitin")


async def test_moss_without_a_server_url_refuses_to_be_built() -> None:
    with pytest.raises(TranscribeError, match="MOSS_URL"):
        MossDriver(FakeHTTP({}), url=None, model="moss-test", deadline=timedelta(seconds=5), max_new_tokens=1024)


async def test_moss_health_asks_the_models_of_the_sidecar(driver: MossDriver, http: FakeHTTP) -> None:
    assert await driver.health() is True
    assert http.gets == ["http://127.0.0.1:1/v1/models"]
