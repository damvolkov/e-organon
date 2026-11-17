"""e_organon.models.audio: the `audio/` tools' inputs and answers — a recording asked, a transcript kept."""

from typing import Annotated

import msgspec

### an ISO 639-1 tag or empty for the engine to decide; kept short as the door's other language hints are.
LANG = Annotated[str, msgspec.Meta(max_length=3)]


##### TRANSCRIBE #####
class TranscribeIn(msgspec.Struct, frozen=True, kw_only=True):
    """`path`: an audio or video file. `language`: a two-letter hint, empty lets the engine hear it. `hotwords`:
    spellings the engine would otherwise guess — names, products, terms of the room. `srt`: also write subtitles."""

    path: str
    language: LANG = ""
    hotwords: tuple[str, ...] = ()
    srt: bool = False


class SegmentOut(msgspec.Struct, frozen=True, kw_only=True):
    """One utterance: its span in seconds, its anonymous speaker (empty when the engine heard one voice), its text."""

    start: float
    end: float
    speaker: str
    text: str


class TranscribeOut(msgspec.Struct, frozen=True, kw_only=True):
    """The kept files and what they hold: the segments themselves ride along, as doc/convert carries its markdown."""

    path: str
    json_path: str
    srt_path: str = ""
    language: str
    duration: float
    seconds: float
    segments: list[SegmentOut]
