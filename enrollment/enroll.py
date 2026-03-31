"""
Speaker enrollment CLI.

Records audio (or loads a WAV file), computes a 256-dim resemblyzer
voiceprint embedding, and registers the speaker in memory-mcp.

Usage
-----
Enroll from microphone (10 seconds):
    python enroll.py --name Brian --room office --duration 10

Enroll from WAV file:
    python enroll.py --name Sarah --wav path/to/sarah.wav

List all enrolled speakers:
    python enroll.py --list

Show unenrolled provisional voices (from live audio, auto-created by pipeline):
    python enroll.py --unknown
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import warnings
import wave
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# webrtcvad (a resemblyzer dependency) uses pkg_resources which is deprecated in
# Python 3.13+ setuptools.  Suppress before the import fires — we cannot patch
# a third-party package.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
    module="webrtcvad",
)

# resemblyzer is optional — not available in test environments without the package.
# Importing at module level makes VoiceEncoder and preprocess_wav patchable in tests.
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:  # pragma: no cover
    VoiceEncoder = None  # type: ignore[assignment,misc]
    preprocess_wav = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VOICEPRINT_DIM = 256
MIN_AUDIO_S = 3.0  # minimum recording duration for a reliable embedding


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    _configure_logging(args.log_level if hasattr(args, "log_level") else "INFO")
    asyncio.run(_dispatch(args))


async def _dispatch(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    import os
    load_dotenv()

    memory_mcp_url = os.getenv("MEMORY_MCP_URL", "http://memory-mcp:8900")
    memory_mcp_token = os.getenv("MEMORY_MCP_TOKEN", "")
    device_index = int(os.getenv("DEVICE_INDEX") or 0)

    headers: dict[str, str] = {}
    if memory_mcp_token:
        headers["Authorization"] = f"Bearer {memory_mcp_token}"

    async with httpx.AsyncClient(base_url=memory_mcp_url, timeout=30.0, headers=headers) as client:
        if args.command == "enroll":
            await cmd_enroll(client, args, device_index)
        elif args.command == "list":
            await cmd_list(client)
        elif args.command == "unknown":
            await cmd_unknown(client)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_enroll(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    device_index: int,
) -> None:
    """Enroll a speaker from mic or WAV file."""
    name: str = args.name
    room: Optional[str] = getattr(args, "room", None)
    duration: float = getattr(args, "duration", 10.0)
    wav_path: Optional[str] = getattr(args, "wav", None)

    print(f"Enrolling speaker: {name}")

    # --- Acquire audio ---
    if wav_path:
        print(f"  Loading audio from: {wav_path}")
        audio = load_wav(wav_path)
    else:
        print(f"  Recording {duration:.0f}s from microphone... (speak now)")
        audio = record_audio(duration_s=duration, device_index=device_index)
        print("  Recording complete.")

    if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
        print(
            f"ERROR: Audio too short ({len(audio)/SAMPLE_RATE:.1f}s). "
            f"Minimum {MIN_AUDIO_S:.0f}s required for a reliable voiceprint.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Compute embedding ---
    print("  Computing voiceprint embedding...")
    embedding = compute_embedding(audio)
    if embedding is None:
        print("ERROR: Failed to compute voiceprint embedding.", file=sys.stderr)
        sys.exit(1)
    print(f"  Embedding computed (norm={float(np.linalg.norm(embedding)):.4f})")

    # --- Register entity in memory-mcp ---
    print(f"  Registering entity '{name}' in memory-mcp...")
    entity_result = await _ensure_entity(client, name, entity_type="person")
    if entity_result is None:
        print("ERROR: Failed to create entity in memory-mcp.", file=sys.stderr)
        sys.exit(1)

    # --- Store voiceprint ---
    print("  Storing voiceprint embedding...")
    vp_result = await _update_voiceprint(client, name, embedding)
    if vp_result is None:
        print("ERROR: Failed to store voiceprint in memory-mcp.", file=sys.stderr)
        sys.exit(1)

    # --- Confirm ---
    profile = await _get_profile(client, name)
    if isinstance(profile, dict):
        meta = (profile.get("result") or {}).get("meta") or {}
        if not meta and isinstance(profile.get("meta"), dict):
            meta = profile["meta"]
        sample_count = meta.get("voiceprint_samples", 1)
        print(f"\nEnrolled: {name}")
        print(f"  Room     : {room or 'not specified'}")
        print(f"  Samples  : {sample_count}")
        print(f"  Status   : {meta.get('status', 'enrolled')}")
    else:
        print(f"\nEnrolled: {name} (voiceprint stored successfully)")


async def cmd_list(client: httpx.AsyncClient) -> None:
    """List all enrolled speakers."""
    try:
        response = await client.get("/entities")
        response.raise_for_status()
        entities = response.json().get("entities", [])
    except Exception as exc:
        print(f"ERROR: Failed to fetch entities: {exc}", file=sys.stderr)
        sys.exit(1)

    enrolled = [
        e for e in entities
        if e.get("type") == "person"
        and (e.get("meta") or {}).get("status") == "enrolled"
    ]

    if not enrolled:
        print("No enrolled speakers found.")
        return

    print(f"\nEnrolled speakers ({len(enrolled)}):")
    print(f"  {'Name':<20} {'Samples':>8}  {'First seen'}")
    print(f"  {'-'*20} {'-'*8}  {'-'*24}")
    for e in enrolled:
        meta = e.get("meta") or {}
        name = e.get("name", "?")
        samples = meta.get("voiceprint_samples", "?")
        first_seen = meta.get("first_seen", "unknown")
        print(f"  {name:<20} {str(samples):>8}  {first_seen}")


async def cmd_unknown(client: httpx.AsyncClient) -> None:
    """List unenrolled provisional voices."""
    try:
        response = await client.get("/voices/unknown", params={"limit": 50})
        response.raise_for_status()
        voices = response.json().get("result", [])
    except Exception as exc:
        print(f"ERROR: Failed to fetch unknown voices: {exc}", file=sys.stderr)
        sys.exit(1)

    if not voices:
        print("No unenrolled voices found.")
        return

    print(f"\nUnenrolled voices ({len(voices)}):")
    print(f"  {'Entity':<30} {'Detections':>10}  {'Sample transcript'}")
    print(f"  {'-'*30} {'-'*10}  {'-'*40}")
    for v in voices:
        name = v.get("entity_name", "?")
        count = v.get("detection_count", 0)
        transcript = (v.get("sample_transcript") or "")[:40]
        print(f"  {name:<30} {count:>10}  {transcript!r}")

    print(
        f"\nTo enroll: python enroll.py --name <Name> --wav <file.wav>"
        f"\nOr use --merge to attach to an existing speaker."
    )


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------


def record_audio(duration_s: float, device_index: int = 0) -> np.ndarray:
    """Record mono 16kHz audio from the specified device.

    Args:
        duration_s: Recording duration in seconds.
        device_index: sounddevice device index.

    Returns:
        float32 mono array normalised to [-1, 1].

    Raises:
        ImportError: If sounddevice is not installed.
        RuntimeError: On recording failure.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for microphone recording. "
            "Install with: pip install sounddevice"
        ) from exc

    # Auto-detect ReSpeaker by name so the correct device is used regardless
    # of index changes across reboots.  Falls back to device_index if not found.
    try:
        device = sd.query_devices("reSpeaker", kind="input")["index"]
    except Exception:
        device = device_index

    # Query the device to determine how many input channels it supports.
    # The ReSpeaker XVF3800 requires 2 channels; standard mics use 1.
    try:
        dev_info = sd.query_devices(device, kind="input")
        channels = min(2, int(dev_info["max_input_channels"]))
    except Exception:
        channels = 1

    n_samples = int(duration_s * SAMPLE_RATE)
    recording = sd.rec(
        n_samples,
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    # Always return mono — take channel 0 (beamformed output on ReSpeaker)
    return recording[:, 0]


def load_wav(path: str) -> np.ndarray:
    """Load a WAV file and return a float32 mono 16kHz array.

    Accepts mono or stereo WAV files at any sample rate; converts to
    mono 16kHz internally.

    Args:
        path: Path to the WAV file.

    Returns:
        float32 mono array at SAMPLE_RATE Hz.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a WAV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(p), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        file_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Parse samples
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    # Convert stereo to mono
    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    elif n_channels > 2:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16kHz if needed
    if file_rate != SAMPLE_RATE:
        try:
            import scipy.signal
            n_out = int(len(samples) * SAMPLE_RATE / file_rate)
            samples = scipy.signal.resample(samples, n_out).astype(np.float32)
        except ImportError:
            raise ImportError(
                f"WAV file is {file_rate} Hz but scipy is required for resampling. "
                "Install with: pip install scipy  OR provide a 16kHz WAV file."
            )

    return samples


# ---------------------------------------------------------------------------
# Voiceprint computation
# ---------------------------------------------------------------------------


def compute_embedding(audio: np.ndarray) -> Optional[np.ndarray]:
    """Compute a 256-dim resemblyzer voiceprint embedding.

    Args:
        audio: float32 mono array at SAMPLE_RATE Hz.

    Returns:
        Unit-norm float32 array of shape (256,), or None on failure.
    """
    if VoiceEncoder is None or preprocess_wav is None:
        raise ImportError(
            "resemblyzer is required for voiceprint computation. "
            "Install with: pip install resemblyzer"
        )
    try:
        encoder = VoiceEncoder()
        wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
        embedding = encoder.embed_utterance(wav).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            return None
        return embedding / norm
    except Exception as exc:
        logger.error("Embedding computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# memory-mcp API helpers
# ---------------------------------------------------------------------------


async def _ensure_entity(
    client: httpx.AsyncClient,
    name: str,
    entity_type: str = "person",
) -> Optional[dict]:
    """Create or update a person entity in memory-mcp.

    Uses POST /remember to ensure the entity exists (memory-mcp creates the
    entity automatically when a fact is stored for a new entity name).

    Args:
        client: Authenticated httpx client.
        name: Entity name to create/confirm.
        entity_type: Entity type string.

    Returns:
        Response dict or None on failure.
    """
    try:
        response = await client.post(
            "/remember",
            json={
                "entity_name": name,
                "entity_type": entity_type,
                "fact": f"{name} is an enrolled speaker in the Heimdall audio pipeline.",
                "category": "enrollment",
                "source": "enrollment_cli",
                "meta": {"status": "enrolled"},
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error("Failed to ensure entity '%s': %s", name, exc)
        return None


async def _update_voiceprint(
    client: httpx.AsyncClient,
    name: str,
    embedding: np.ndarray,
) -> Optional[dict]:
    """POST the voiceprint embedding to memory-mcp /voices/update_print.

    Args:
        client: Authenticated httpx client.
        name: Entity name.
        embedding: 256-dim float32 unit-norm array.

    Returns:
        Response dict or None on failure.
    """
    try:
        response = await client.post(
            "/voices/update_print",
            json={
                "entity_name": name,
                "embedding": embedding.tolist(),
                "weight": 1.0,  # first enrollment — take the new embedding fully
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error("Failed to update voiceprint for '%s': %s", name, exc)
        return None


async def _get_profile(
    client: httpx.AsyncClient,
    name: str,
) -> Optional[dict]:
    """GET entity profile for confirmation display.

    Args:
        client: Authenticated httpx client.
        name: Entity name.

    Returns:
        Profile dict or None if not found.
    """
    try:
        response = await client.get(f"/profile/{name}")
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enroll",
        description="Enroll speakers into the Heimdall voice identity system.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # enroll sub-command
    enroll_parser = sub.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("--name", required=True, help="Speaker name, e.g. 'Brian'")
    enroll_parser.add_argument(
        "--room", default=None, help="Room where enrollment was recorded (metadata only)"
    )
    enroll_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Microphone recording duration in seconds (default: 10)",
    )
    enroll_parser.add_argument(
        "--wav",
        default=None,
        metavar="PATH",
        help="Path to a WAV file to use instead of mic recording",
    )

    # list sub-command
    sub.add_parser("list", help="List all enrolled speakers")

    # unknown sub-command
    sub.add_parser("unknown", help="List unenrolled provisional voice entities")

    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


if __name__ == "__main__":
    main()
