#!/usr/bin/env python3
"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight -- record the camera microphone and plot its power spectrum.

Finds the camera's USB microphone, records 10 seconds (configurable) from every
channel it exposes, and saves:

    <output-dir>/audio_<time>.wav      raw samples, one WAV channel per USB channel
    <output-dir>/spectrum_<time>.png   power spectrum of each channel

It also prints RMS level, peak level and the strongest frequency of each channel.
The samples are what the camera delivers: no mixing, filtering or resampling.
What you do with them is up to you; this file is only a starting point.

Usage:
    python example_audio.py                      # 10 s from the camera microphone
    python example_audio.py --duration 5
    python example_audio.py --list               # show the input devices and exit
    python example_audio.py --device 3           # pick a device by index (or name part)

Ctrl+C stops the recording early; what was recorded so far is still saved.
Exit codes: 0 ok, 2 no microphone found, 4 the audio stream failed, 130 Ctrl+C.
"""

import argparse
import sys
from pathlib import Path

from audio_utils import (
    EXPECTED_CHANNELS,
    AudioDeviceNotFoundError,
    AudioNotAvailableError,
    AudioStreamError,
    channel_stats,
    compute_power_spectrum,
    find_audio_device,
    format_devices,
    list_audio_devices,
    record_audio,
    save_power_spectrum_plot,
    save_wav,
    timestamp,
)
from camera_utils import install_stop_flag


def run(args):
    stop = install_stop_flag()
    try:
        if args.list:
            print(format_devices(list_audio_devices()))
            return 0
        info = find_audio_device(args.device)
        print(f"Microphone: [{info['index']}] {info['name']} ({info['hostapi']}), "
              f"{info['max_input_channels']} channel(s)")
        if info["max_input_channels"] != EXPECTED_CHANNELS and not args.channels:
            print(f"  NOTE: this firmware sends audio on {EXPECTED_CHANNELS} channels; "
                  f"if the OS offers more ({info['max_input_channels']} here), the extra "
                  f"channels are expected to be silent. Use --channels {EXPECTED_CHANNELS} "
                  f"to record only the real ones.")
        print(f"Recording {args.duration:g} s (Ctrl+C to stop early) ...")
        audio, fs, stats = record_audio(args.duration, info, args.samplerate, args.channels,
                                        stop)
    except (AudioDeviceNotFoundError, AudioNotAvailableError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except AudioStreamError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    if len(audio) < 2:
        print("Nothing was recorded.")
        return 130 if stats["interrupted"] else 4

    out = Path(args.output_dir)
    stamp = timestamp()
    freqs, psd_db = compute_power_spectrum(audio, fs)
    wav = save_wav(out / f"audio_{stamp}.wav", audio, fs)
    png = save_power_spectrum_plot(
        freqs, psd_db, out / f"spectrum_{stamp}.png",
        title=f"Power spectrum, {audio.shape[1]} channel(s), {fs} Hz, "
              f"{stats['duration_actual_s']:.1f} s")

    print(f"\n{audio.shape[1]} channel(s) at {fs} Hz, {stats['duration_actual_s']:.2f} s, "
          f"{stats['xruns']} buffer over/underruns")
    print("channel   rms (dBFS)   peak (dBFS)   strongest (Hz)")
    for s in channel_stats(audio, fs, freqs, psd_db):
        note = ""
        if s["silent"]:
            note = "  all zero!"
        elif s["duplicate_of"]:
            note = f"  identical to channel {s['duplicate_of']}"
        print(f"{s['channel']:>7}   {s['rms_dbfs']:10.1f}   {s['peak_dbfs']:11.1f}   "
              f"{s['dominant_hz']:14.0f}{note}")
    print(f"\nSaved {wav}\nSaved {png}")
    return 130 if stats["interrupted"] else 0


def main():
    p = argparse.ArgumentParser(description="LogiInsight -- record the camera microphone "
                                            "and plot its power spectrum")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Seconds to record (default: 10)")
    p.add_argument("--device", type=str, default=None,
                   help="Input device index or part of its name (default: auto-detect)")
    p.add_argument("--samplerate", type=int, default=None,
                   help="Sample rate in Hz (default: 48000, else the device default)")
    p.add_argument("--channels", type=int, default=None,
                   help="Number of channels (default: all the device exposes)")
    p.add_argument("--output-dir", type=str, default="audio_captures",
                   help="Where to write the .wav and .png (default: audio_captures)")
    p.add_argument("--list", action="store_true", help="List the input devices and exit")
    args = p.parse_args()
    if args.duration <= 0:
        p.error("--duration must be > 0")
    print("LogiInsight microphone example")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
