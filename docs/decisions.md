# Decisions

Recorded here: what was hard to reverse, surprising without context, and the result of a real trade-off. Date, evidence,
rejected alternatives. Everything else lives in the code and the changelog.

## 2026-11-17 — `audio/transcribe`: MOSS-Transcribe-Diarize 0.9B as the speech engine

**Context.** Long-form, multi-speaker transcription, local on one RTX 4090, Spanish first-hand use. The queued corpus:
~60-minute meeting recordings.

**Decision.** MOSS-Transcribe-Diarize 0.9B (OpenMOSS), served by sglang-omni behind the OpenAI audio endpoint.

**Evidence.** Open ASR Leaderboard mean WER 5.17 across the published set and RTFx 294; single pass to ~90 min with
timestamps and anonymous speaker labels from one model (no VAD + clustering pipeline to stitch); 1st place, 2nd MLC-SLM
Challenge (INTERSPEECH 2026) across 14 languages including Spanish; Apache-2.0; ~2 GB bf16 weights, ~18 GB container at
`--mem-fraction-static 0.75`.

**Rejected.** *Parakeet TDT v3* (best EN-only dictation, no diarization — would need pyannote on top, two models, one
pipeline to own); *WhisperX* (whisper + wav2vec align + pyannote: four stages, weak on Spanish names, worse long-form
WER than the leaderboard leaders); *running the weights in-process* (organon would carry torch: the house API service
stays light, and the slot bound is the queue's, not a GPU held twice).

## 2026-11-17 — the speech server is a driver sidecar, not an adapter

**Decision.** The model lives in its own container; `MossDriver` speaks HTTP (`Locality.SIDECAR`, `Workload.GPU_BOUND`)
through the shared `HTTPPort`. vLLM's audio endpoint would answer the same port without touching the driver.

**Rejected.** *Adapter* (transformers or faster-whisper in-process): violates the GPU law of the house — one card, one
owner, bounded by the queue slot (`ORGANON_SLOTS_AUDIO=1`), no torch in the application image.

## 2026-11-17 — GPU layout: the speech sidecar owns the card alone while it serves (Damien: on-demand)

**Context.** The 4090 is the LLM slot: vLLM (Gemma 26B AWQ) at `--gpu-memory-utilization 0.86` plus TEI hold 23.7/24.5
GiB; MOSS is single-pass (`allow_audio_chunking` is False for it on sglang-omni) — its encoder materializes the whole
recording (~1.4 GiB gelu intermediates at 61 min) and the KV holds audio plus transcript to the last token (~20 k
tokens per dense minute).

**Decision.** `~/.config/compose/moss` sits behind the compose profile `speech`: to transcribe, the whole GPU stack
out (`stop vllm embed reranker`) → `up moss` (`--mem-fraction-static 0.62`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
→ jobs → down → stack back. `MOSS_URL` is permanent in organon's env: with the sidecar down the provider answers
`api_status` as unhealthy and the tool refuses; up, it serves.

**Evidence (bench, `.ergon/reports/plans/transcribe.md`).** 0.75 and 0.65 with TEI resident: CUDA OOM in the encoder.
0.50 with TEI resident: completes but the dense 62-min meeting ran KV at 0.97 — no margin. 0.62 exclusive: KV 0.62,
RTFx 24.4 (sparse) / 17.0 (dense) on the real meetings — an hour in two-three minutes, startup 112–175 s. The vendor's
RTFx 294 is a batched harness on a clean card; the pool at 0.62 (~116 k tokens) is what makes the 90-min door honest.

**Rejected.** *Coexistence with TEI/vLLM resident* (measured above — the pool the model needs and the margin the
encoder needs cannot both fit); *lowering the door below 90 min* (the exclusive window fits dense 90-min meetings);
*multi-model vLLM* (MOSS support is nightly-pinned, the house image is a release tag).
