"""e_organon.mcp.tools.audio: the audio/ section as tools — the twin of `api/router/audio`, whose very handlers they are.

The transcription tool rides only where its sidecar does: a deployment without `MOSS_URL` never sees `audio_transcribe`
in the catalog, as it never sees `doc_ocr` without its engine."""

from typing import Final

from e_organon.api.router import audio as router
from e_organon.infra.auth.models import Access
from e_organon.infra.mcp.models import MCPTool
from e_organon.mcp.refusals import REFUSALS

##### TOOLS — what a client is told #####
TRANSCRIBE: Final = MCPTool(
    name="audio_transcribe",
    title="Transcribe a recording",
    description=(
        "An audio or video file by `path`, whole, into a timestamped speaker-aware transcript: `language` (two-letter "
        "hint, empty hears it), `hotwords` (spellings the engine would guess: names, terms) and `srt` for subtitles. "
        "The segments come back with the answer; the Markdown, JSON (and SRT) land in the outbox as `path`, `json_path`, "
        "`srt_path`. Up to 90 min in one pass. Long jobs: ask with organon_submit and read with organon_job."
    ),
    read_only=False,
)


##### ROUTES #####
### served only where the sidecar answers (MOSS_URL).
ROUTES: Final = (
    TRANSCRIBE.route(Access.USER, dependencies=router.TOOL_DEPS, exception_handlers=REFUSALS)(router.transcribe),
)
