"""One reusable sensor implementation instantiated independently for six stations."""

from datetime import datetime
from uuid import UUID

from sensorlab.dataset import Observation
from sensorlab.registry import STATION_SENSORS
from sensorlab.telemetry import TelemetryMessage


class VirtualSensor:
    def __init__(self, station: str, session_id: UUID):
        self.sensor_id = STATION_SENSORS[station]
        self.station = station
        self.session_id = session_id
        self.transmitted = 0
        self.last_observation: datetime | None = None

    def prepare(self, observation: Observation, sent_at: datetime) -> TelemetryMessage:
        if observation.station != self.station:
            raise ValueError("observation belongs to another sensor")
        if self.last_observation is not None and observation.time <= self.last_observation:
            raise ValueError("sensor observation timestamps must strictly increase")
        return TelemetryMessage(
            schema_version="1.0",
            sensor_id=self.sensor_id,
            station=self.station,
            session_id=self.session_id,
            sequence=self.transmitted + 1,
            observation_time=observation.time,
            sent_at=sent_at,
            measurements=observation.measurements,
        )

    def commit(self, message: TelemetryMessage) -> None:
        if (
            message.station != self.station
            or message.sensor_id != self.sensor_id
            or message.session_id != self.session_id
            or message.sequence != self.transmitted + 1
        ):
            raise ValueError("cannot commit a foreign or out-of-sequence message")
        if self.last_observation is not None and message.observation_time <= self.last_observation:
            raise ValueError("cannot commit a non-increasing observation timestamp")
        self.transmitted += 1
        self.last_observation = message.observation_time

    def status(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "station": self.station,
            "transmitted": self.transmitted,
            "next_sequence": self.transmitted + 1,
            "last_observation": self.last_observation.isoformat()
            if self.last_observation
            else None,
        }
