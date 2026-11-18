"""e_organon.models.audio: the `audio/` tools' inputs and answers — a recording asked, a transcript kept."""

from typing import Annotated

import msgspec

### an ISO 639-1 tag or empty for the engine to decide; kept short as the door's other language hints are.
LANG = Annotated[str, msgspec.Meta(max_length=3)]


##### TRANSCRIBE #####
class TranscribeIn(msgspec.Struct, frozen=True, kw_only=True):
    """`path`: an audio or video file. `language`: a two-letter hint, empty lets the engine hear it. `hotwords`:
    spellings the engine would otherwise guess — names, products, terms of the room. `srt`: also write subtitles.
    `names`: name each speaker from the video, where it shows who speaks (a meeting's grid); off, labels only."""

    path: str
    language: LANG = ""
    hotwords: tuple[str, ...] = ()
    srt: bool = False
    names: bool = True


class SegmentOut(msgspec.Struct, frozen=True, kw_only=True):
    """One utterance: its span in seconds, its anonymous speaker (empty when the engine heard one voice), its text, and
    the speaker's name when the video showed it (empty otherwise)."""

    start: float
    end: float
    speaker: str
    text: str
    name: str = ""


class TranscribeOut(msgspec.Struct, frozen=True, kw_only=True):
    """The kept files and what they hold: the segments themselves ride along, as doc/convert carries its markdown.
    `windows`: how many requests the engine answered (one per window, a re-asked tail included); `covered`: where the
    last segment ends, in seconds — short of `duration` only by the silence at the end; `speakers`: each label the video
    named, and its name."""

    path: str
    json_path: str
    srt_path: str = ""
    language: str
    duration: float
    seconds: float
    windows: int = 1
    covered: float = 0.0
    speakers: dict[str, str] = msgspec.field(default_factory=dict)
    segments: list[SegmentOut]
