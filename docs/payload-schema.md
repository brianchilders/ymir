# AudioPayload — Canonical Wire Format

## Purpose

`AudioPayload` is the JSON structure posted by every room node to the pipeline
worker at `POST /ingest`.  It is the single canonical interface between the
edge (Pi) and the backend (blackmagic.lan).

Two node profiles share the same schema — fields are optional depending on what
the node computed locally.

Defined in: `pipeline_worker/models.py`

---

## Node Profiles

| Capability | Full Node (Pi 5 + Hailo-8) | Capture Node (Pi 4) |
|---|---|---|
| `transcript` | Whisper small/medium output | `null` |
| `whisper_confidence` | [0, 1] score | `null` |
| `emotion` | valence/arousal | `null` |
| `voiceprint` | 256-dim embedding | `null` |
| `audio_clip_b64` | Only when confidence < threshold | Always present |
| `node_profile` | `"full"` | `"capture"` |

---

## Schema

### Full node (high-confidence utterance)

```json
{
  "room":               "kitchen",
  "timestamp":          "2026-03-22T10:30:00Z",
  "node_profile":       "full",
  "transcript":         "I need to pick up groceries tomorrow",
  "whisper_confidence": 0.92,
  "whisper_model":      "small",
  "doa":                247,
  "emotion": {
    "valence":  0.6,
    "arousal":  0.3
  },
  "voiceprint":         [0.012, -0.034, ...],
  "audio_clip_b64":     null,
  "duration_ms":        null
}
```

### Full node (low-confidence — audio clip attached)

Same as above but `audio_clip_b64` is populated and `duration_ms` is set.
`transcript` is still present as a hint to the pipeline worker.

### Capture node

```json
{
  "room":               "bedroom",
  "timestamp":          "2026-03-22T10:30:00Z",
  "node_profile":       "capture",
  "doa":                90,
  "audio_clip_b64":     "<base64-encoded WAV>",
  "duration_ms":        2500
}
```

All inference fields (`transcript`, `whisper_confidence`, `emotion`,
`voiceprint`) are absent or `null`.

---

## Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `room` | string | yes | Room identifier, e.g. `kitchen`, `bedroom` |
| `timestamp` | ISO-8601 UTC | yes | Start time of the utterance |
| `node_profile` | `"full"` \| `"capture"` | no (default: `"full"`) | Node capability profile |
| `transcript` | string | no | Whisper transcript (min 1 char, stripped); `null` for capture nodes |
| `whisper_confidence` | float [0,1] | no | Mean segment log-prob converted to probability; `null` for capture nodes |
| `whisper_model` | string | no | Model variant: `small`, `medium`, `large-v3` (default: `small`) |
| `doa` | int [0,359] | no | Direction of arrival in degrees; `null` if unavailable |
| `emotion.valence` | float [0,1] | no | 0=very negative, 1=very positive; `null` for capture nodes |
| `emotion.arousal` | float [0,1] | no | 0=calm, 1=excited; `null` for capture nodes |
| `voiceprint` | float[256] | no | Resemblyzer GE2E embedding; `null` for capture nodes or short clips |
| `audio_clip_b64` | string | no | Base64 mono 16kHz PCM WAV; always present for capture nodes |
| `duration_ms` | int | no | Clip duration in ms; present whenever `audio_clip_b64` is set |

---

## Pipeline Worker Routing

```python
async def ingest(payload: AudioPayload):
    if payload.node_profile == "capture" or payload.audio_clip_b64:
        # Run full inference stack on blackmagic
        transcript, confidence = await whisper_large_v3.transcribe(payload.audio_clip_b64)
        voiceprint = await resemblyzer.embed(payload.audio_clip_b64)
        emotion = await emotion_model.predict(payload.audio_clip_b64)
    else:
        # Trust full node's on-device results
        transcript = payload.transcript
        ...
```

---

## Validation Rules

- `transcript` is stripped of leading/trailing whitespace; empty string after stripping → 422
- `transcript` may be `null` (capture nodes do not produce transcripts)
- `whisper_confidence` must be [0.0, 1.0] if present
- `node_profile` must be `"full"` or `"capture"`
- `doa` must be [0, 359] if present
- `voiceprint` must be exactly 256 floats if present
- `audio_clip_b64` must be valid base64 if present
- `duration_ms` must be >= 0 if present
- `emotion.valence` and `emotion.arousal` must be [0.0, 1.0]

---

## Memory-mcp Storage Schema

After identity resolution, the pipeline worker writes a `voice_activity`
reading to Tier 2 with this composite value:

```json
{
  "entity_name": "Brian",
  "metric": "voice_activity",
  "value": {
    "transcript":         "I need to pick up groceries tomorrow",
    "confidence":         0.92,
    "doa":                247,
    "room":               "kitchen",
    "whisper_model":      "small",
    "whisper_confidence": 0.92,
    "speaker_confidence": 0.91,
    "emotion": {
      "valence": 0.6,
      "arousal": 0.3
    }
  },
  "source": "audio_pipeline"
}
```
