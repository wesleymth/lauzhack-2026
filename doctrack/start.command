#!/bin/bash
# Launch DocTrack (double-click in Finder, or the Start button of the Logi MX keypad plugin).
# Opens in Terminal on purpose: Terminal already has camera + microphone access, and the log
# (transcription, export, errors) stays visible.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
printf '\033]0;DocTrack\007'            # Terminal window title
echo "Starting DocTrack ..."
exec uv run python -m doctrack.app "$@"
