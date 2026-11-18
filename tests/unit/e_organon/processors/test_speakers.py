"""tests/unit/e_organon/processors/speakers: windows stitched under one set of labels (linked by the seconds two windows
spoke alike in their overlap, cut at its middle, the overlap ending where the earlier window stopped hearing), and labels
named by the video — one ring per picture, a strip read once per place while it looks alike, names cleaned, folded and
voted."""

from typing import TYPE_CHECKING

import msgspec
import pytest

from e_organon.processors.speakers import Seen, SpeakersProcessor, Window
from support.media import pictures
from support.ocr import said
from support.stage import ring
from support.transcribe import seg

if TYPE_CHECKING:
    from e_organon.ports.transcribe import Segment
    from support.ocr import FakeOcr
    from support.stage import FakeStage

PLACE = (0, 0, 64, 36)
ELSEWHERE = (64, 0, 64, 36)
WHOLE = Window(start=0, end=100, segments=(seg(0, 100, "S01", "a"),))


def _reading(out: tuple[Segment, ...]) -> list[tuple[str, str]]:
    return [(each.speaker, each.text) for each in out]


def _window(start: float, end: float, *segments: Segment) -> Window:
    return Window(start=start, end=end, segments=segments)


##### WINDOW #####


@pytest.mark.parametrize(
    ("window", "heard"),
    [
        (_window(0, 100, seg(0, 40, "S01"), seg(40, 70, "S02")), 70),
        (_window(0, 100, seg(0, 40, "S01"), seg(90, 120, "S02")), 100),
        (_window(30, 100), 30),
    ],
    ids=["last-segment", "never-past-the-end", "nothing-heard"],
)
async def test_window_heard_is_where_the_last_segment_ends_within_the_window(window: Window, heard: float) -> None:
    assert window.heard == heard


##### STITCH #####


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        (
            [
                _window(
                    0,
                    100,
                    seg(0, 40, "S01", "a"),
                    seg(40, 80, "S02", "b"),
                    seg(80, 90, "", "c"),
                    seg(90, 99, "S01", "d"),
                )
            ],
            [("S01", "a"), ("S02", "b"), ("", "c"), ("S01", "d")],
        ),
        ([_window(0, 100, seg(0, 40, "S09", "a"), seg(40, 80, "S04", "b"))], [("S01", "a"), ("S02", "b")]),
        (
            [WHOLE, _window(50, 150, seg(50, 70, "S01", "b1"), seg(80, 120, "S01", "b2"))],
            [("S01", "a"), ("S01", "b2")],
        ),
        (
            [WHOLE, _window(50, 150, seg(50, 51, "S01", "b1"), seg(110, 120, "S01", "b2"))],
            [("S01", "a"), ("S02", "b2")],
        ),
        (
            [WHOLE, _window(50, 150, seg(50, 52, "S01", "b1"), seg(110, 120, "S01", "b2"))],
            [("S01", "a"), ("S01", "b2")],
        ),
        (
            [
                _window(0, 100, seg(0, 60, "S01", "a1"), seg(60, 80, "S02", "a2"), seg(80, 100, "S03", "a3")),
                _window(50, 150, seg(50, 75, "S01", "b1"), seg(76, 100, "S01", "b2")),
            ],
            [("S01", "a1"), ("S02", "a2"), ("S04", "b2")],
        ),
        (
            [
                _window(0, 100, seg(0, 60, "S01", "a1"), seg(60, 100, "S02", "a2")),
                _window(50, 150, seg(60, 75, "S01", "b1"), seg(76, 90, "S01", "b2"), seg(90, 100, "S02", "b3")),
            ],
            [("S01", "a1"), ("S02", "a2"), ("S02", "b2"), ("S03", "b3")],
        ),
        (
            [
                _window(0, 100, seg(0, 100, "S01", "a"), seg(60, 70, "", "a0")),
                _window(50, 150, seg(55, 70, "", "b1"), seg(76, 90, "", "b2"), seg(80, 120, "S01", "b3")),
            ],
            [("S01", "a"), ("", "a0"), ("", "b2"), ("S01", "b3")],
        ),
        (
            [
                _window(0, 100, seg(0, 55, "S01", "a1"), seg(90, 100, "S02", "a2")),
                _window(50, 150, seg(60, 70, "S01", "b1"), seg(76, 85, "S01", "b2")),
            ],
            [("S01", "a1"), ("S03", "b2")],
        ),
        (
            [_window(0, 40, seg(0, 40, "S01", "a")), _window(50, 100, seg(50, 90, "S01", "b"))],
            [("S01", "a"), ("S02", "b")],
        ),
        (
            [
                _window(0, 100, seg(0, 40, "S01", "a")),
                _window(30, 100, seg(30, 34, "S01", "b1"), seg(36, 60, "S01", "b2")),
            ],
            [("S01", "a"), ("S01", "b2")],
        ),
        (
            [
                _window(0, 100, seg(0, 100, "S01", "a")),
                _window(90, 190, seg(90, 100, "S01", "b1"), seg(105, 190, "S02", "b2")),
                _window(180, 280, seg(180, 190, "S01", "c1"), seg(195, 280, "S01", "c2")),
            ],
            [("S01", "a"), ("S02", "b2"), ("S02", "c2")],
        ),
    ],
    ids=[
        "one-window-keeps-its-labels",
        "one-window-numbers-by-first-appearance",
        "linked",
        "below-link-is-a-new-speaker",
        "exactly-link-links",
        "below-lead-share-is-a-new-speaker",
        "largest-overlap-chooses-first-a-label-is-taken-once",
        "empty-speaker-untouched",
        "no-speech-at-once-no-link",
        "no-overlap-numbers-on",
        "overlap-ends-where-the-earlier-window-stopped",
        "links-chain-through-the-relabelled-window",
    ],
)
async def test_speakers_stitch_links_labels_and_cuts_the_overlap(
    windows: list[Window], expected: list[tuple[str, str]]
) -> None:
    assert _reading(SpeakersProcessor.stitch(windows)) == expected


async def test_speakers_stitch_sorts_windows_given_out_of_order() -> None:
    windows = [WHOLE, _window(50, 150, seg(50, 70, "S01", "b1"), seg(80, 120, "S02", "b2"))]
    assert SpeakersProcessor.stitch(windows[::-1]) == SpeakersProcessor.stitch(windows)


##### HEARD #####


def _called(start: float, end: float, speaker: str, name: str = "") -> Segment:
    return msgspec.structs.replace(seg(start, end, speaker), name=name)


def _who(out: tuple[Segment, ...]) -> list[tuple[float, str, str]]:
    return [(each.start, each.speaker, each.name) for each in out]


async def test_speakers_stitch_links_one_name_across_windows_that_share_no_overlap() -> None:
    first = _window(0, 100, _called(0, 50, "S01", "Ana"), _called(50, 90, "S02"))
    ### the later window heard Ana only after the overlap: the name alone says she is S01.
    later = _window(80, 200, _called(80, 95, "S07"), _called(150, 190, "S03", "Ana"))
    assert _who(SpeakersProcessor.stitch([first, later])) == [
        (0, "S01", "Ana"),
        (50, "S02", ""),
        (150, "S01", "Ana"),
    ]


async def test_speakers_stitch_never_links_two_names_however_much_they_share() -> None:
    first = _window(0, 100, _called(0, 50, "S01", "Ana"), _called(50, 100, "S02", "Luis"))
    ### in the overlap the later label spoke over Luis for 40 s, but it is Marta: a third speaker, not Luis.
    later = _window(60, 200, _called(60, 100, "S05", "Marta"), _called(100, 200, "S05", "Marta"))
    out = SpeakersProcessor.stitch([first, later])
    assert {(each.speaker, each.name) for each in out} == {("S01", "Ana"), ("S02", "Luis"), ("S03", "Marta")}


async def test_speakers_stitch_names_every_segment_of_a_label_named_in_any_window() -> None:
    ### linked by the overlap, unnamed in the first window, named in the second: the whole label is Ana.
    first = _window(0, 100, _called(0, 30, "S01"), _called(40, 100, "S02"))
    later = _window(60, 200, _called(60, 100, "S04"), _called(120, 200, "S04", "Ana"))
    out = SpeakersProcessor.stitch([first, later])
    assert [(each.start, each.speaker, each.name) for each in out if each.speaker == "S02"] == [
        (40, "S02", "Ana"),
        (120, "S02", "Ana"),
    ]


async def test_speakers_heard_counts_only_the_pictures_with_exactly_one_ring(
    speakers: SpeakersProcessor, stage: FakeStage, ocr: FakeOcr
) -> None:
    ocr.readings = {ring(PLACE, 1).strip: said("Ana García")}
    stage.told = {1.0: (ring(PLACE, 1),), 3.0: (ring(PLACE, 1), ring(ELSEWHERE, 1))}
    seen = await speakers.heard(pictures(1.0, 2.0, 3.0))
    assert (seen, stage.seen) == ((Seen(at=1.0, name="Ana García"),), [1.0, 2.0, 3.0])


async def test_speakers_heard_reads_a_strip_once_per_place_while_it_looks_alike(
    speakers: SpeakersProcessor, stage: FakeStage, ocr: FakeOcr
) -> None:
    ocr.readings = {ring(PLACE, 1).strip: said("Ana García"), ring(PLACE, 2).strip: said("Luis Pérez")}
    stage.told = {
        1.0: (ring(PLACE, 1, look=0),),
        2.0: (ring(PLACE, 1, look=5),),
        3.0: (ring(PLACE, 2, look=100),),
        4.0: (ring(ELSEWHERE, 1, look=0),),
    }
    seen = await speakers.heard(pictures(1.0, 2.0, 3.0, 4.0))
    ### the second look is within `alike` of the first: not read; the third is not; the fourth is another place.
    assert [each.name for each in seen] == ["Ana García", "Ana García", "Luis Pérez", "Ana García"]
    assert [language for _, _, language in ocr.seen] == ["spa"] * 3


async def test_speakers_heard_an_empty_reading_is_not_kept_and_names_nobody(
    speakers: SpeakersProcessor, stage: FakeStage, ocr: FakeOcr
) -> None:
    stage.told = {1.0: (ring(PLACE, 1),), 2.0: (ring(PLACE, 1),)}
    assert (await speakers.heard(pictures(1.0, 2.0)), len(ocr.seen)) == ((), 2)


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        (("Ana García (Allianz)",), "Ana García"),
        (("Ana García Allianz",), "Ana García"),
        (("Ana García Alianz",), "Ana García"),
        (("Ana García Llianz",), "Ana García"),
        (("Ana", "García", "(Allianz)"), "Ana García"),
        (("  Ana   García  ",), "Ana García"),
        ((", Ana García -",), "Ana García"),
        (("Ana García..",), "Ana García"),
        (("Ana Garc…",), "Ana Garc"),
        (("(Allianz)",), ""),
        (("..",), ""),
    ],
    ids=[
        "brackets",
        "allianz",
        "alianz",
        "llianz",
        "words-joined",
        "spaces-folded",
        "edges-stripped",
        "dots-cut",
        "ellipsis-cut",
        "only-the-organisation",
        "only-dots",
    ],
)
async def test_speakers_heard_cleans_the_name_off_the_organisation(
    speakers: SpeakersProcessor, stage: FakeStage, ocr: FakeOcr, raw: tuple[str, ...], name: str
) -> None:
    ocr.readings = {ring(PLACE, 1).strip: said(*raw)}
    stage.told = {1.0: (ring(PLACE, 1),)}
    assert [each.name for each in await speakers.heard(pictures(1.0))] == ([name] if name else [])


##### NAME #####


def _named(speakers: SpeakersProcessor, segments: list[Segment], seen: list[Seen]) -> list[tuple[str, str]]:
    return [(each.speaker, each.name) for each in speakers.name(segments, seen)]


def _sighted(name: str, *at: float) -> list[Seen]:
    return [Seen(at=instant, name=name) for instant in at]


async def test_speakers_name_gives_a_label_the_name_seen_while_it_spoke(speakers: SpeakersProcessor) -> None:
    segments = [seg(0, 4, "S01"), seg(0, 4, ""), seg(5, 9, "S02"), seg(10, 12, "S03")]
    ### the span's own ends count: a sighting at 0 and one at 4 are inside seg(0, 4).
    seen = _sighted("Ana García", 0.0, 2.0, 4.0)
    assert _named(speakers, segments, seen) == [("S01", "Ana García"), ("", ""), ("S02", ""), ("S03", "")]


async def test_speakers_name_with_nothing_seen_changes_nothing(speakers: SpeakersProcessor) -> None:
    segments = [seg(0, 4, "S01")]
    assert tuple(speakers.name(segments, [])) == tuple(segments)


@pytest.mark.parametrize(
    ("seen", "expected"),
    [
        (_sighted("Ana García", 1.0, 2.0), ""),
        (_sighted("Ana García", 20.0, 21.0, 22.0), ""),
        (
            sorted(
                [*_sighted("Luis Pérez", 0.5, 0.6, 0.7), *_sighted("Ana García", 1.0, 2.0, 3.0)], key=lambda s: s.at
            ),
            "",
        ),
        (_sighted("Ana García", 1.0, 2.0) + _sighted("Ana Garcia", 3.0), "Ana García"),
        (_sighted("Ana García", 1.0, 2.0) + _sighted("Ana García 1", 3.0), "Ana García"),
        (_sighted("Lu", 1.0, 2.0, 3.0), ""),
        (_sighted("ab1 cde2345678", 1.0, 2.0, 3.0), ""),
        (_sighted("1234", 1.0, 2.0, 3.0), ""),
    ],
    ids=[
        "too-few-votes",
        "seen-outside-the-span",
        "share-not-held",
        "variants-fold-onto-the-most-frequent",
        "a-variant-that-resembles-folds-too",
        "no-word-of-three-letters",
        "mostly-digits",
        "only-digits",
    ],
)
async def test_speakers_name_needs_votes_a_share_and_a_name_that_reads_as_one(
    speakers: SpeakersProcessor, seen: list[Seen], expected: str
) -> None:
    assert _named(speakers, [seg(0, 4, "S01")], seen) == [("S01", expected)]


@pytest.mark.parametrize(
    ("seen", "expected"),
    [
        (_sighted("Diego", 12, 14), "Diego"),
        (_sighted("Diego", 12), ""),
        ([*_sighted("Diego", 12, 14), *_sighted("Angel", 16)], ""),
        ([], ""),
    ],
    ids=["one-name-twice", "seen-once", "two-names", "nothing-seen"],
)
async def test_speakers_each_names_an_unnamed_segment_by_the_one_name_shown_while_it_lasted(
    speakers: SpeakersProcessor, seen: list[Seen], expected: str
) -> None:
    (out,) = speakers.each([seg(10, 18, "S07")], seen)
    assert out.name == expected


async def test_speakers_each_leaves_a_named_segment_as_it_is(speakers: SpeakersProcessor) -> None:
    (out,) = speakers.each([_called(10, 18, "S07", "Ana")], _sighted("Diego", 12, 14, 16))
    assert out.name == "Ana"
