"""infra/config/services: the services cluster — organon's own knobs, one `### NAME` block per service.

UPPERCASE typed fields, the field name the environment variable itself, read once at the composition root and handed
down as plain arguments."""

from datetime import timedelta
from pathlib import Path
from typing import Annotated

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from e_organon.infra.config.base import BaseSettings


##### SETTINGS #####
class ServiceSettings(BaseSettings):
    """The canonical tools' configuration."""

    ##### DATA #####

    ### DATA — organon's own state, one subtree per section (`doc/models/tessdata`…): what a section downloads lands
    ### here once and is reused by every later start. The image sets `/data`, a volume.
    ORGANON_DATA: Path = Path("data")

    ##### FILES #####

    ### ROOTS — where a tool's path argument may point: the vault (mounted at its own path in every harness, Hermes'
    ### upload cache included) and the workspaces.
    ORGANON_ROOTS: tuple[Path, ...] = (Path("~/.config/memory"), Path("~/workspaces"))
    ### OUTBOX — where a tool's output files land, content-addressed: inside a root, so an output is the next input.
    ORGANON_OUTBOX: Path = Path("~/.config/memory/workflow/organon")

    ##### LIMITS #####

    ### MAX_PAGES — the most pages a PDF may have before a tool opens it: a page tree can name one page twenty thousand
    ### times in a few hundred kilobytes, and render, merge and split walk it page by page.
    ORGANON_MAX_PAGES: PositiveInt = 5_000

    ### MAX_AUDIO — the longest recording `audio/transcribe` accepts, read from the header before anything is decoded:
    ### the engine answers a single pass of this much; longer lives are someone else's problem (cut it, then ask).
    ORGANON_MAX_AUDIO: timedelta = timedelta(minutes=90)

    ##### AUDIO #####

    ### WINDOW — a recording longer than this is heard in windows of it, one request each. The sidecar holds 65 536
    ### tokens for audio and transcript together: an hour of audio is ~44 800 of them (12.2 a second), so one pass of
    ### a talkative hour ran out of room at 37 of 61 minutes and answered success (measured 2026-11-18). Thirty minutes
    ### leave the transcript ~43 000: room to spare, and windows long enough to see each speaker's face at least once.
    ORGANON_AUDIO_WINDOW: timedelta = timedelta(minutes=30)
    ### OVERLAP — how much two neighbouring windows both hear: where their speakers are linked and their texts cut.
    ORGANON_AUDIO_OVERLAP: timedelta = timedelta(minutes=2)
    ### RETRIES — how many times a window that stopped hearing before its end is asked again for what it dropped.
    ORGANON_AUDIO_RETRIES: NonNegativeInt = 2
    ### STAGE — how often a video is looked at for who is speaking, and the size it is looked at.
    ORGANON_STAGE_EVERY: timedelta = timedelta(seconds=2)
    ORGANON_STAGE_WIDTH: PositiveInt = 1280
    ORGANON_STAGE_HEIGHT: PositiveInt = 720
    ### a name strip already read, looked at again: the mean grey difference under which it is the same strip.
    ORGANON_STAGE_ALIKE: PositiveFloat = 10.0
    ### a label is named when its name was seen this many times at least, and in this share of its sightings.
    ORGANON_STAGE_VOTES: PositiveInt = 3
    ORGANON_STAGE_SHARE: Annotated[float, Field(gt=0.5, le=1.0)] = 0.6
    ### a segment its label could not name is named on its own when one name, and only it, was seen this many times
    ### while it lasted (a label the engine gave two voices).
    ORGANON_STAGE_ALONE: PositiveInt = 2

    ##### IMAGES #####

    ### MAX_PIXELS — the most pixels an image tool decodes (width x height, read from the header first): decoding
    ### allocates in proportion, so a photo over it is refused unread. 40 MP: a 200 MP phone shot is not a signature.
    ORGANON_MAX_PIXELS: PositiveInt = 40_000_000

    ##### JOBS #####

    ### SLOTS — how many jobs of a section run at once, fleet-wide on the durable queue; one per section that has jobs.
    ORGANON_SLOTS_DOC: PositiveInt = 2
    ORGANON_SLOTS_IMAGE: PositiveInt = 2
    ### one at a time: the speech sidecar is one model on one GPU, and an hour of audio is not a small thing to feed it.
    ORGANON_SLOTS_AUDIO: PositiveInt = 1

    ##### TEKTON #####

    ### PROFILES — the nomos of the bridge, a YAML book read once at startup: what a caller may make a coder do, by name.
    ### Unset, the bridge is off: no coder, no `tekton/` tools — the public image runs the tools alone.
    TEKTON_PROFILES: Path | None = None
    ### WORKSPACES — the root a project's short name is looked up under (`e-voice` → `<root>/<owner>/e-voice`).
    TEKTON_WORKSPACES: Path = Path("~/workspaces")
    ### LIMIT — how many coder runs may be in flight at once; the rest wait their turn.
    TEKTON_LIMIT: PositiveInt = 4
    ### CLAUDE — the Claude Code CLI the coder runs and reaches live sessions through (a name on PATH, or a path).
    TEKTON_CLAUDE: str = "claude"
