# speech: local speech-to-text with timestamps

Records from any input device and transcribes it locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small.en`, int8, CPU).
Every sentence gets a start and end time, and every word gets a start `time`.
Audio and transcript are written **while you record**, so long sessions are safe.

## TL;DR

```bash
cd speech
uv sync                          # first time only (the model downloads on the first run)
uv run python stt.py --list      # show microphones
uv run python stt.py             # record from the default mic; Ctrl+C to stop
uv run python stt.py --device 0  # or choose a mic by number or name (--camera for LogiInsight)
```

Results go to `sessions/<date_time>/transcript.json` (text and timestamps) and `session.wav`.
On macOS, allow microphone access for your terminal or VS Code.

## Setup

```bash
cd speech
uv sync
```

The model (~250 MB) downloads on the first run. After that everything runs offline.
On macOS, give your terminal microphone access (System Settings → Privacy & Security → Microphone).

## Usage

```bash
uv run python stt.py --list              # list input devices
uv run python stt.py                     # system default input
uv run python stt.py --camera            # LogiInsight camera mic (auto-detected)
uv run python stt.py --device 3          # by index from --list
uv run python stt.py --device "MacBook"  # by part of the name
```

Speak. Sentences are printed, and `transcript.json` is updated, about every 25 s:
the audio is cut into ~25 s chunks at a quiet moment and transcribed in the background.
Press **Ctrl+C** to stop. The last part is transcribed, then saved.

```
[00:00.7–00:03.0] The mitochondria is the powerhouse of the cell.
[00:03.4–00:08.2] It produces most of the chemical energy needed to power the cell's biochemical reactions.
Saved sessions/2026-11-21_20-35-26/ (17 sentences, 59.8 s audio)
```

## Output: `sessions/<date_time>/`

- `session.wav` is mono 16-bit audio at the device's native rate. It is written continuously.
- `transcript.json` is rewritten atomically after every chunk, so other programs can read it during recording.

```json
{
 "session_start_epoch": 1763753726.412,
 "sentences": [
  {"start": 0.7, "end": 3.0, "text": "The mitochondria is the powerhouse of the cell.",
   "words": [{"word": "The", "time": 0.7}, {"word": "mitochondria", "time": 0.86}, "..."]}
 ]
}
```

Times are seconds since the session started. Wall-clock time is `session_start_epoch + time`.

```python
import json
t = json.load(open("sessions/2026-11-21_20-35-26/transcript.json"))
for s in t["sentences"]:
    print(s["start"], s["end"], s["text"])
    for w in s["words"]:
        print("  ", w["time"], w["word"])
```

Speed: 60 s of audio takes about 8 s to transcribe on an Apple Silicon CPU.
Device lookup reuses `../logiinsight_api/audio_utils.py`.
