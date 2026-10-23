"""Local sinks and extension interfaces; network delivery remains outside this project."""

from typing import Protocol, TextIO

from pydantic import BaseModel

from sensorlab.telemetry import TelemetryMessage


class TelemetrySink(Protocol):
    """A future HTTP adapter may implement this boundary without changing sensors."""

    def write(self, message: TelemetryMessage) -> None: ...


class MessageSigner(Protocol):
    """Sign exact serialized bytes; Ed25519 implements this without changing playback."""

    def sign(self, payload: bytes) -> bytes: ...


class NDJSONSink:
    def __init__(self, *streams: TextIO):
        if not streams:
            raise ValueError("at least one output stream is required")
        self.streams = streams

    def write(self, message: BaseModel) -> None:
        line = message.model_dump_json() + "\n"
        for stream in self.streams:
            stream.write(line)
            stream.flush()
