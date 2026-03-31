# Heimdall Troubleshooting

---

## ReSpeaker XVF3800 — USB Audio I/O Errors on Pi 4

**Symptom:** `arecord: pcm_read:2272: read error: Input/output error` and `retire_capture_urb: NNNNN callbacks suppressed` in dmesg.

**Cause:** The Pi 4's black USB 2.0 ports cannot sustain the isochronous USB audio stream from the XVF3800.

**Fix:** Plug the ReSpeaker into a **blue USB 3.0 port**. USB 2.0 devices work fine in USB 3.0 ports and the xHCI controller handles the isochronous transfers reliably.

**Verify:**
```bash
arecord -D hw:1,0 -f S16_LE -r 16000 -c 2 -d 3 /tmp/test.wav && echo "OK"
```

---

## ReSpeaker XVF3800 — DOA Access Denied (Errno 13)

**Symptom:** `DOA read failed: [Errno 13] Access denied (insufficient permissions)` in capture node logs.

**Cause:** The HID device node for the ReSpeaker is owned by root and not readable by the service user.

**Fix:** Add a udev rule to set permissions on the HID device:
```bash
echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="001a", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/99-respeaker.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Restart the capture node — `DOA read failed` should no longer appear.

---

## ReSpeaker XVF3800 — Device Not Found (Wrong PID)

**Symptom:** `ReSpeaker USB device not found (VID=0x2886, tried PIDs: 0x0018, 0x0019, 0x0020, 0x002b)`

**Cause:** The XVF3800 4-Mic Array uses PID `0x001a`, which was not in the original PID list.

**Fix:** Already patched in `room_node/doa.py`. Verify the device PID with:
```bash
lsusb | grep -i seeed
```

---

## ReSpeaker — Channels Count Not Available at 1 Channel

**Symptom:** `arecord: set_params:1398: Channels count non available` when recording with `-c 1`.

**Cause:** The XVF3800 requires 2 channels (stereo). Mono capture is not supported directly.

**Fix:** Always record with `-c 2`. The beamformed audio is on channel 0; `capture.py` takes only channel 0 from the stereo stream.
```bash
arecord -D hw:1,0 -f S16_LE -r 16000 -c 2 -d 3 /tmp/test.wav
```

---

## Silero VAD — "Input audio chunk is too short"

**Symptom:**
```
builtins.ValueError: Input audio chunk is too short
```

**Cause:** Silero VAD requires a minimum of 512 samples per chunk. The original 30ms chunk (480 samples at 16kHz) is below this minimum.

**Fix:** Already patched in `room_node/capture.py` — `CHUNK_SAMPLES = 512`.

---

## Pipeline Worker — ModuleNotFoundError: No module named 'pipeline_worker'

**Symptom:** Running `python -m uvicorn pipeline_worker.server:app` from inside the `pipeline_worker/` directory fails with `ModuleNotFoundError`.

**Cause:** The module path `pipeline_worker.server` requires the project root on `sys.path`. Running from inside the subdirectory puts the wrong directory on the path.

**Fix:** Always run from the project root:
```bash
cd ~/projects/heimdall
python -m uvicorn pipeline_worker.server:app --host 0.0.0.0 --port 8001
```

---

## Pipeline Worker — `.env` Not Loaded (Settings Show Defaults)

**Symptom:** Pipeline worker logs show `memory-mcp: http://localhost:8900` even though `.env` has the correct URL.

**Cause:** `pydantic-settings` resolves `.env` relative to the working directory. If the `.env` is in the project root but you run from a subdirectory (or vice versa), it won't be found.

**Fix:** `settings.py` now resolves `.env` relative to the module file first, then falls back to cwd. Keep `.env` in the project root or in `pipeline_worker/` — either location works.

---

## Capture Node — Ctrl+C Does Not Exit

**Symptom:** Pressing Ctrl+C prints the shutdown message but the process hangs and never exits.

**Cause:** `iter_utterances()` blocks on `queue.get()` in a background thread. After `capture.stop()` closes the audio stream, the thread remains blocked waiting for a chunk that never arrives.

**Fix:** Already patched in `room_node/capture.py` — `stop()` now pushes a `None` sentinel into the queue to unblock the iterator.

---

## sounddevice DEVICE_INDEX — Does Not Match ALSA Card Number

**Symptom:** Wrong device used for capture, or `Invalid device` error.

**Cause:** sounddevice assigns its own index numbers that don't always match ALSA card numbers.

**Fix:** Always find the correct index with:
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```
Use the number in brackets at the start of the ReSpeaker line as `DEVICE_INDEX` in `.env`.

---

## Python Version — torch/torchaudio CUDA Error on Pi

**Symptom:**
```
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```

**Cause:** `pip install torch` without specifying an index URL may pull a CUDA build even on ARM.

**Fix:** Always install torch with the CPU-only index on Pi:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```
