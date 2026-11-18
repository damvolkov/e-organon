"""processors/speakers: one speaker, one label, across a recording heard in windows — and a name for it when the video
shows who is speaking. Deterministic: no model is asked anything here.

**Stitching.** A long recording is heard in overlapping windows, and the speech engine diarizes each on its own: its
`S01` in one window is nobody in particular in the next. Two things say who is who across windows, in this order. A
name, when each window was named first: a label of the later window named as a speaker already heard is that speaker.
Then the overlap, which both windows heard: a label takes the earlier window's label it shares the most speaking time
with there — when that is at least `LINK` seconds and most of what the label said in the overlap, no other label of the
window took it first (the largest overlaps choose first), and the two do not carry different names. Two names never
merge: an overlap that pairs them is noise, and a wrong link would fuse two people into one label that no vote can then
name. A label nothing links is a new speaker, numbered on. A label named in any window names all its segments. The two
readings of the overlap are cut at its middle, the earlier keeping what starts before the cut. The overlap ends where
the earlier window stopped hearing — its last segment —, not where it was meant to end: a window re-asked for what an
engine dropped overlaps only what was really heard.

**Naming.** A call app marks the speaking tile (`StagePort`) and writes the person's name under it; an OCR reads the
strip. A strip already read at the same place, looking alike (a mean grey difference under `alike`), is not read
again: video noise is no new name. Only a picture with exactly one marked tile says who spoke at that instant. What the
OCR read is cut to the person (the organisation after them goes), its variants folded onto the most frequent spelling
they resemble (`difflib`, ratio `FOLD`), and what reads as no name dropped. Each label then counts the names seen while
it spoke: the most seen names it when seen at least `votes` times and in at least `share` of its sightings. A label
spoken only under a shared screen — no grid, no tile — keeps no name: it is not guessed."""

import bisect
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Final

import msgspec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

    from e_organon.ports.media import Frame
    from e_organon.ports.ocr import OcrPort
    from e_organon.ports.stage import Ring, StagePort
    from e_organon.ports.transcribe import Segment

### the least shared speaking time, in seconds, that links a label across an overlap.
LINK: Final = 2.0
### the share of a label's overlap time its link must hold.
LEAD: Final = 0.5
### two spellings of one name: difflib's ratio between them, lowercased.
FOLD: Final = 0.8
### what follows the person in a Teams label: the organisation in brackets, or — where the OCR lost the bracket — its
### first word, or the ellipsis of a cut label.
AFTER: Final = re.compile(r"\s*(\(|\b[a-z]?l+ia?nz\b|\.\.|…).*$", re.IGNORECASE)
SPACES: Final = re.compile(r"\s+")
### a name has a word of at least three letters, and is mostly letters.
WORD: Final = re.compile(r"[^\W\d_]{3,}")
LETTERS: Final = 0.6
### the edges stripped off a name: punctuation an OCR leaves where the label was cut.
EDGES: Final = " ,.-·"
### the label of a speaker the stitching numbers on.
LABEL: Final = "S{:02d}"


##### VALUES #####
@dataclass(frozen=True, slots=True, kw_only=True)
class Window:
    """One window heard: where it was meant to start and end in the recording, and its segments in the recording's
    time, labelled as the engine labelled them."""

    start: float
    end: float
    segments: tuple[Segment, ...]

    @property
    def heard(self) -> float:
        """Where the window stopped hearing: its last segment's end, never past where it was meant to end."""
        return min(self.end, max((each.end for each in self.segments), default=self.start))


@dataclass(frozen=True, slots=True, kw_only=True)
class Seen:
    """One instant the video showed one speaking tile, and the name read under it."""

    at: float
    name: str


##### PROCESSOR #####
class SpeakersProcessor:
    """Windows stitched under one set of labels; labels named by what the video showed."""

    __slots__ = ("_alike", "_height", "_language", "_ocr", "_share", "_stage", "_votes", "_width")

    def __init__(
        self,
        stage: StagePort,
        ocr: OcrPort,
        *,
        width: int,
        height: int,
        alike: float,
        votes: int,
        share: float,
        language: str = "spa",
    ) -> None:
        self._stage, self._ocr, self._width, self._height = stage, ocr, width, height
        self._alike, self._votes, self._share, self._language = alike, votes, share, language

    ##### PRIVATE #####

    @staticmethod
    def _stitch_shared(
        earlier: Sequence[Segment], later: Sequence[Segment], start: float, end: float
    ) -> dict[tuple[str, str], float]:
        """Seconds each (later label, earlier label) pair spoke at once inside the overlap `[start, end)`, summed."""
        inside = [each for each in earlier if each.speaker and each.end > start and each.start < end]
        shared: defaultdict[tuple[str, str], float] = defaultdict(float)
        for b in later:
            for a in inside if b.speaker and b.end > start and b.start < end else ():
                shared[b.speaker, a.speaker] += max(0.0, min(b.end, a.end, end) - max(b.start, a.start, start))
        return {pair: seconds for pair, seconds in shared.items() if seconds > 0}

    @staticmethod
    def _stitch_names(segments: Iterable[Segment]) -> dict[str, str]:
        """Each label's name, where its segments carry one: the name it spoke longest under."""
        spoken: defaultdict[str, Counter] = defaultdict(Counter)
        for each in segments:
            _ = each.speaker and each.name and spoken[each.speaker].update({each.name: each.end - each.start})
        return {label: counted.most_common(1)[0][0] for label, counted in spoken.items()}

    @classmethod
    def _stitch_link(
        cls, earlier: Sequence[Segment], later: Sequence[Segment], start: float, end: float
    ) -> dict[str, str]:
        """Each later label to the earlier label that is the same speaker: by name first, then by the overlap they
        share, never across two different names."""
        named, called = cls._stitch_names(earlier), cls._stitch_names(later)
        by_name = {name: label for label, name in sorted(named.items(), reverse=True)}
        links = {label: by_name[name] for label, name in called.items() if name in by_name}
        shared = cls._stitch_shared(earlier, later, start, end)
        spoken: defaultdict[str, float] = defaultdict(float)
        for (b, _), seconds in shared.items():
            spoken[b] += seconds
        taken: set[str] = set(links.values())
        for (b, a), seconds in sorted(shared.items(), key=lambda item: (-item[1], item[0])):
            apart = b in called and a in named and called[b] != named[a]
            match (b in links, a in taken, apart, seconds >= LINK and seconds >= LEAD * spoken[b]):
                case (False, False, False, True):
                    links[b], _ = a, taken.add(a)
        return links

    @staticmethod
    def _stitch_relabel(window: Window, links: Mapping[str, str], used: list[str]) -> tuple[Segment, ...]:
        """The window's segments under the recording's labels: linked ones take their link, the rest are numbered on."""
        names: dict[str, str] = dict(links)
        for label in dict.fromkeys(each.speaker for each in window.segments if each.speaker):
            match names.get(label):
                case None:
                    names[label] = LABEL.format(len(used) + 1)
                    used.append(names[label])
        return tuple(
            msgspec.structs.replace(each, speaker=names[each.speaker]) if each.speaker else each
            for each in window.segments
        )

    @staticmethod
    def _name_clean(raw: str) -> str:
        """The person in a label: the organisation after them gone, spaces folded, cut edges stripped."""
        return SPACES.sub(" ", AFTER.sub("", raw)).strip(EDGES)

    @staticmethod
    def _name_valid(name: str) -> bool:
        """A name reads as one: a word of three letters at least, and mostly letters."""
        letters = sum(each.isalpha() for each in name)
        return bool(WORD.search(name)) and letters >= LETTERS * len(name.replace(" ", ""))

    @classmethod
    def _name_fold(cls, counted: Counter) -> dict[str, str]:
        """Each spelling to the most frequent one it resembles, the most frequent first; one that reads as no name and
        resembles none is dropped."""
        canon: list[str] = []
        folded: dict[str, str] = {}
        for name, _ in sorted(counted.items(), key=lambda item: (-item[1], item[0])):
            match next(
                (each for each in canon if SequenceMatcher(None, name.lower(), each.lower()).ratio() >= FOLD), None
            ):
                case None if cls._name_valid(name):
                    canon.append(name)
                    folded[name] = name
                case None:
                    pass
                case found:
                    folded[name] = found
        return folded

    def _heard_alike(self, look: bytes, other: bytes) -> bool:
        """Two looks of a strip within the mean grey difference that is still the same strip."""
        return sum(abs(a - b) for a, b in zip(look, other, strict=True)) < self._alike * len(look)

    async def _heard_read(self, ring: Ring, read: dict[tuple[int, int, int, int], list[tuple[bytes, str]]]) -> str:
        """The name in one ring's strip: a strip already read at this place that looks alike, else the OCR's reading.
        A reading that found nothing is not kept: the next picture tries again."""
        held = read.setdefault(ring.place, [])
        match next((said for look, said in held if self._heard_alike(ring.look, look)), None):
            case str() as said:
                return said
            case _:
                said = " ".join(each.text for each in await self._ocr.read(ring.strip, self._language)).strip()
                _ = said and held.append((ring.look, said))
                return said

    ############################################################

    ##### PUBLIC #####

    @classmethod
    def stitch(cls, windows: Sequence[Window]) -> tuple[Segment, ...]:
        """The windows as one transcript under one set of labels, in the recording's order; segments already named (each
        window named on its own, `name`) carry their name to their label."""
        ordered = sorted(windows, key=lambda each: each.start)
        used: list[str] = []
        kept: list[Segment] = []
        previous: Window | None = None
        for window in ordered:
            match previous:
                case None:
                    relabelled = cls._stitch_relabel(window, {}, used)
                    kept.extend(relabelled)
                case before:
                    end = before.heard
                    cut = (window.start + end) / 2 if end > window.start else window.start
                    links = cls._stitch_link(kept, window.segments, window.start, end)
                    relabelled = cls._stitch_relabel(window, links, used)
                    kept = [each for each in kept if each.start < cut]
                    kept.extend(each for each in relabelled if each.start >= cut)
            previous = Window(start=window.start, end=window.end, segments=relabelled)
        names = cls._stitch_names(kept)
        return tuple(
            msgspec.structs.replace(each, name=names[each.speaker]) if each.speaker in names else each for each in kept
        )

    async def heard(self, frames: AsyncIterator[Frame]) -> tuple[Seen, ...]:
        """Every instant one tile was marked as speaking, and the name under it, cleaned; instants with none, or more
        than one, say nothing."""
        read: dict[tuple[int, int, int, int], list[tuple[bytes, str]]] = {}
        seen: list[Seen] = []
        async for frame in frames:
            match await self._stage.rings(frame, width=self._width, height=self._height):
                case (ring,):
                    match self._name_clean(await self._heard_read(ring, read)):
                        case "":
                            pass
                        case name:
                            seen.append(Seen(at=frame.at, name=name))
        return tuple(seen)

    def name(self, segments: Iterable[Segment], seen: Sequence[Seen]) -> tuple[Segment, ...]:
        """The segments with their speaker's name, where the video named that speaker clearly."""
        whole = tuple(segments)
        folded = self._name_fold(Counter(each.name for each in seen))
        sightings = [(each.at, folded[each.name]) for each in seen if each.name in folded]
        instants = [at for at, _ in sightings]
        votes: dict[str, Counter] = defaultdict(Counter)
        for each in whole:
            for _, name in sightings[
                bisect.bisect_left(instants, each.start) : bisect.bisect_right(instants, each.end)
            ]:
                votes[each.speaker][name] += 1
        named = {
            label: top
            for label, counted in votes.items()
            if label
            for top, count in counted.most_common(1)
            if count >= self._votes and count >= self._share * counted.total()
        }
        return tuple(
            msgspec.structs.replace(each, name=named[each.speaker]) if each.speaker in named else each for each in whole
        )
