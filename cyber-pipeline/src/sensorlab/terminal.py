"""Portable interactive controls kept separate from the playback implementation."""

import json
import os
import sys
from threading import Event, Thread
from typing import TextIO

from sensorlab.playback import PlaybackEngine


def dispatch_control(command: str, engine: PlaybackEngine) -> dict:
    key = command.strip().lower()
    if key in {"p", "pause"}:
        engine.pause()
    elif key in {"r", "resume"}:
        engine.resume()
    elif key in {"q", "quit", "stop"}:
        engine.stop()
    elif key not in {"s", "status", ""}:
        raise ValueError("controls: p = pause, r = resume, s = status, q = stop")
    return engine.status()


class TerminalControls:
    def __init__(self, engine: PlaybackEngine, output: TextIO = sys.stderr):
        self.engine = engine
        self.output = output
        self._done = Event()
        self._thread = Thread(target=self._read, name="sensorlab-controls", daemon=True)

    def start(self) -> None:
        self.output.write("Controls: p pause | r resume | s status | q stop (Unix: press Enter)\n")
        self.output.flush()
        self._thread.start()

    def close(self) -> None:
        self._done.set()
        self._thread.join(timeout=0.3)

    def _read(self) -> None:
        while not self._done.wait(0.05):
            if os.name == "nt":
                import msvcrt

                if not msvcrt.kbhit():
                    continue
                command = msvcrt.getwch()
            else:
                import select

                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if not ready:
                    continue
                command = sys.stdin.readline()
                if not command:
                    return
            try:
                status = dispatch_control(command, self.engine)
            except ValueError as exc:
                self.output.write(f"{exc}\n")
            else:
                self.output.write(json.dumps(status) + "\n")
            self.output.flush()
