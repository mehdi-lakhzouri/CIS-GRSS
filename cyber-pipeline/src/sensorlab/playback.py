"""One deterministic chronological queue and six independent sensor sequence states."""

import math
from enum import StrEnum
from threading import Condition, RLock
from uuid import UUID, uuid4

from sensorlab.clock import Clock, SystemClock
from sensorlab.dataset import DetectorDataset
from sensorlab.registry import STATION_SENSORS
from sensorlab.sensor import VirtualSensor
from sensorlab.transport import TelemetrySink


class PlaybackState(StrEnum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class PlaybackEngine:
    """Blocking runner with thread-safe pause/resume/stop/status controls.

    Scheduling uses monotonic active time, excluding pauses. Equal observation times
    have deterministic station order. Wall time supplies sent_at only.
    """

    def __init__(
        self,
        dataset: DetectorDataset,
        sink: TelemetrySink,
        speed: float = 3600,
        clock: Clock | None = None,
        session_id: UUID | None = None,
    ):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("speed must be finite and >= 0 (0 means no-wait)")
        self.dataset = dataset
        self.sink = sink
        self.speed = speed
        self.clock = clock or SystemClock()
        self.session_id = session_id or uuid4()
        self.sensors = {
            station: VirtualSensor(station, self.session_id) for station in STATION_SENSORS
        }
        self._condition = Condition(RLock())
        self._state = PlaybackState.READY
        self._started = 0.0
        self._paused_at = 0.0
        self._pause_duration = 0.0
        self._error: str | None = None
        self._has_run = False
        self._selected_stations = dataset.stations

    def pause(self) -> bool:
        with self._condition:
            if self._state != PlaybackState.RUNNING:
                return False
            self._paused_at = self.clock.monotonic()
            self._state = PlaybackState.PAUSED
            self._condition.notify_all()
            return True

    def resume(self) -> bool:
        with self._condition:
            if self._state != PlaybackState.PAUSED:
                return False
            self._pause_duration += self.clock.monotonic() - self._paused_at
            self._state = PlaybackState.RUNNING
            self._condition.notify_all()
            return True

    def stop(self) -> bool:
        with self._condition:
            if self._state not in {
                PlaybackState.READY,
                PlaybackState.RUNNING,
                PlaybackState.PAUSED,
            }:
                return False
            self._state = PlaybackState.STOPPED
            self._condition.notify_all()
            return True

    def _await_due(self, due: float) -> bool:
        while True:
            with self._condition:
                while self._state == PlaybackState.PAUSED:
                    self._condition.wait()
                if self._state != PlaybackState.RUNNING:
                    return False
                active = self.clock.monotonic() - self._started - self._pause_duration
                remaining = due - active
                if remaining <= 0 or self.speed == 0:
                    return True
            # Bounded polling permits interrupting even 1x, hour-long gaps promptly.
            self.clock.sleep(min(remaining, 0.05))

    def run(self) -> dict:
        with self._condition:
            if self._has_run:
                raise RuntimeError(
                    "a playback engine can be started only once; create a new session"
                )
            self._has_run = True
            if self._state == PlaybackState.STOPPED:
                return self.status()
            self._state = PlaybackState.RUNNING
            self._started = self.clock.monotonic()
        first_time = None
        try:
            for observation in self.dataset.observations():
                if first_time is None:
                    first_time = observation.time
                due = (
                    (observation.time - first_time).total_seconds() / self.speed
                    if self.speed
                    else 0.0
                )
                if not self._await_due(due):
                    break
                with self._condition:
                    # A control can arrive after _await_due, before emission.
                    while self._state == PlaybackState.PAUSED:
                        self._condition.wait()
                    if self._state != PlaybackState.RUNNING:
                        break
                    sensor = self.sensors[observation.station]
                    message = sensor.prepare(observation, self.clock.now())
                    self.sink.write(message)
                    sensor.commit(message)
            with self._condition:
                if self._state in {PlaybackState.RUNNING, PlaybackState.PAUSED}:
                    self._state = PlaybackState.COMPLETED
        except KeyboardInterrupt:
            self.stop()
        except Exception as exc:
            with self._condition:
                self._state = PlaybackState.FAILED
                self._error = str(exc)
            raise
        return self.status()

    def status(self) -> dict:
        with self._condition:
            return {
                "state": self._state.value,
                "session_id": str(self.session_id),
                "speed": self.speed,
                "transmitted": sum(sensor.transmitted for sensor in self.sensors.values()),
                "error": self._error,
                "sensors": [
                    {
                        **sensor.status(),
                        "selected": station in self._selected_stations,
                        "state": self._state.value
                        if station in self._selected_stations
                        else "inactive",
                    }
                    for station, sensor in self.sensors.items()
                ],
            }
