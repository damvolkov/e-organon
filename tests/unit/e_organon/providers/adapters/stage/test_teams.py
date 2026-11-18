"""tests/unit/e_organon/providers/adapters/stage/teams: Teams' speaking ring found in pictures drawn here — a closed
outline of the blue-violet, large enough and not a filled patch, its name strip cut as a PNG with a coarse look; what is
no ring (filled, small, open on a side, a shared screen) answers none; what is no picture, or no encodable strip, is a
`MediaError`."""

import cv2
import numpy as np
import pytest

from e_organon.infra.errors import MediaError
from e_organon.ports.media import Frame
from e_organon.providers.adapters.stage.teams import LOOK, STRIP_BOTTOM, STRIP_SIDE, STRIP_TOP, TeamsStageAdapter
from support.stage import SIDES, grid

WIDTH, HEIGHT = 640, 360
TILE = (40, 30, 200, 120)
OTHER = (300, 30, 200, 120)
FOOT = (40, 240, 200, 120)


def _frame(pixels: bytes) -> Frame:
    return Frame(at=0.0, pixels=pixels)


@pytest.fixture
def adapter() -> TeamsStageAdapter:
    return TeamsStageAdapter()


async def test_teams_finds_a_closed_ring_and_cuts_its_name_strip(adapter: TeamsStageAdapter) -> None:
    (found,) = await adapter.rings(_frame(grid(WIDTH, HEIGHT, (TILE, SIDES))), width=WIDTH, height=HEIGHT)
    strip = cv2.imdecode(np.frombuffer(found.strip, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert found.place == TILE
    assert getattr(strip, "shape", None) == (STRIP_TOP - STRIP_BOTTOM, TILE[2] - 2 * STRIP_SIDE, 3)
    ### the name band is white: the look is as many grey bytes as LOOK says, and its brightest is the band's.
    assert (len(found.look), max(found.look)) == (LOOK[0] * LOOK[1], 255)


async def test_teams_finds_every_ring_at_once(adapter: TeamsStageAdapter) -> None:
    rings = await adapter.rings(_frame(grid(WIDTH, HEIGHT, (TILE, SIDES), (OTHER, SIDES))), width=WIDTH, height=HEIGHT)
    assert sorted(each.place for each in rings) == [TILE, OTHER]


async def test_teams_a_tile_running_off_the_picture_needs_no_bottom_side(adapter: TeamsStageAdapter) -> None:
    open_foot = ("top", "left", "right")
    rings = await adapter.rings(_frame(grid(WIDTH, HEIGHT, (FOOT, open_foot))), width=WIDTH, height=HEIGHT)
    assert [each.place for each in rings] == [FOOT]


@pytest.mark.parametrize(
    ("tile", "sides", "filled"),
    [
        (TILE, SIDES, True),
        ((40, 30, 100, 60), SIDES, False),
        (TILE, ("left", "right", "bottom"), False),
        (TILE, ("top", "right", "bottom"), False),
        (TILE, ("top", "left", "bottom"), False),
        (TILE, ("top", "left", "right"), False),
        (TILE, (), False),
    ],
    ids=["filled", "small", "no-top", "no-left", "no-right", "no-bottom-mid-picture", "none-drawn"],
)
async def test_teams_what_is_no_ring_answers_none(
    adapter: TeamsStageAdapter, tile: tuple[int, int, int, int], sides: tuple[str, ...], filled: bool
) -> None:
    picture = grid(WIDTH, HEIGHT, (tile, sides), filled=filled)
    assert await adapter.rings(_frame(picture), width=WIDTH, height=HEIGHT) == ()


async def test_teams_a_picture_of_the_wrong_size_is_refused(adapter: TeamsStageAdapter) -> None:
    with pytest.raises(MediaError, match="640 by 360"):
        await adapter.rings(_frame(b"too short"), width=WIDTH, height=HEIGHT)


async def test_teams_a_strip_that_cannot_be_encoded_is_refused(
    adapter: TeamsStageAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, None))
    with pytest.raises(MediaError, match="could not be encoded"):
        await adapter.rings(_frame(grid(WIDTH, HEIGHT, (TILE, SIDES))), width=WIDTH, height=HEIGHT)


async def test_teams_is_a_library_with_nothing_to_open(adapter: TeamsStageAdapter) -> None:
    async with adapter as opened:
        assert (opened is adapter, await adapter.health()) == (True, True)
