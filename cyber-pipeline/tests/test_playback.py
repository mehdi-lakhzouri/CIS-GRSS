import io
import json
from datetime import timedelta
from threading import Event, Thread
from uuid import UUID

import pytest
from pydantic import ValidationError

from sensorlab.playback import PlaybackEngine
from sensorlab.registry import MEASUREMENT_COLUMNS, STATION_SENSORS
from sensorlab.telemetry import TelemetryMessage
from sensorlab.terminal import dispatch_control
from sensorlab.transport import NDJSONSink

SESSION = UUID("11111111-1111-4111-8111-111111111111")


def test_six_independent_sensors_exact_order_and_reproducibility(dataset, clock, sink):
    engine = PlaybackEngine(dataset, sink, speed=0, clock=clock, session_id=SESSION)
    status = engine.run()
    assert status["state"] == "completed"
    assert status["transmitted"] == 18
    expected = [(row.time, row.station) for row in dataset.observations()]
    assert [(m.observation_time, m.station) for m in sink.messages] == sorted(expected)
    assert len(engine.sensors) == 6
    for station in STATION_SENSORS:
        messages = [m for m in sink.messages if m.station == station]
        assert [m.sequence for m in messages] == [1, 2, 3]
        assert all(m.sensor_id == STATION_SENSORS[station] for m in messages)
        assert all(m.sent_at == clock.origin and m.session_id == SESSION for m in messages)
    stream = io.StringIO()
    PlaybackEngine(dataset, NDJSONSink(stream), speed=0, clock=clock, session_id=SESSION).run()
    assert [json.loads(line) for line in stream.getvalue().splitlines()] == [
        message.model_dump(mode="json") for message in sink.messages
    ]
    with pytest.raises(RuntimeError):
        engine.run()


@pytest.mark.parametrize("speed", [1, 60, 3600, 0])
def test_acceleration_and_sent_at(dataset, clock, sink, speed):
    PlaybackEngine(dataset, sink, speed=speed, clock=clock).run()
    expected = 7200 / speed if speed else 0
    assert clock.elapsed == pytest.approx(expected)
    assert sink.messages[-1].sent_at == clock.origin + timedelta(seconds=expected)
    if not speed:
        assert clock.sleeps == []


def test_pause_resume_without_loss_or_duplication(dataset, clock, sink):
    paused, release = Event(), Event()
    engine = PlaybackEngine(dataset, sink, speed=3600, clock=clock)
    errors = []

    def after_first(message):
        if message.station == next(iter(STATION_SENSORS)) and message.sequence == 1:
            assert engine.pause()
            assert not engine.pause()
            paused.set()

    def worker():
        try:
            engine.run()
        except (AssertionError, RuntimeError, ValueError, OSError) as exc:
            errors.append(exc)
        finally:
            release.set()

    sink.on_write = after_first
    thread = Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert paused.wait(3)
        assert engine.status()["state"] == "paused"
        assert len(sink.messages) == 1
        clock.elapsed += 100  # Paused wall/monotonic time must not advance the schedule.
        assert engine.resume()
        assert not engine.resume()
        assert release.wait(3)
    finally:
        engine.stop()
        thread.join(timeout=3)
    assert not errors
    assert engine.status()["state"] == "completed"
    assert len(sink.messages) == 18
    assert len({(m.station, m.sequence) for m in sink.messages}) == 18
    assert clock.elapsed == pytest.approx(102)


@pytest.mark.parametrize("pause_first", [False, True])
def test_stop_during_emission_or_pause(dataset, clock, sink, pause_first):
    engine = PlaybackEngine(dataset, sink, speed=0, clock=clock)

    def stop(message):
        if len(sink.messages) == 4:
            if pause_first:
                engine.pause()
            engine.stop()

    sink.on_write = stop
    status = engine.run()
    assert status["state"] == "stopped"
    assert status["transmitted"] == len(sink.messages) == 4
    assert not engine.resume()


def test_controls_interrupt_long_wait(dataset, clock, sink):
    engine = PlaybackEngine(dataset, sink, speed=1, clock=clock)
    clock.on_sleep = lambda: dispatch_control("q", engine)
    status = engine.run()
    assert status["state"] == "stopped"
    assert status["transmitted"] == 6
    assert clock.elapsed <= 0.05


def test_sink_failure_does_not_commit_unsent_sequence(dataset, clock, sink):
    engine = PlaybackEngine(dataset, sink, speed=0, clock=clock)

    def fail(message):
        raise OSError("disk full")

    sink.write = fail
    with pytest.raises(OSError, match="disk full"):
        engine.run()
    assert engine.status()["state"] == "failed"
    assert engine.status()["transmitted"] == 0


@pytest.mark.parametrize("speed", [-1, float("nan"), float("inf")])
def test_invalid_speed(dataset, sink, speed):
    with pytest.raises(ValueError):
        PlaybackEngine(dataset, sink, speed=speed)


def test_nulls_roundtrip_and_no_leakage(dataset, clock):
    stream = io.StringIO()
    PlaybackEngine(dataset, NDJSONSink(stream), speed=0, clock=clock).run()
    lines = stream.getvalue().splitlines()
    assert "NaN" not in stream.getvalue()
    assert "Infinity" not in stream.getvalue()
    assert sum(json.loads(line)["measurements"]["P_CALC"] is None for line in lines) == 1
    for line in lines:
        message = TelemetryMessage.model_validate_json(line)
        assert set(message.measurements.model_dump()) == set(MEASUREMENT_COLUMNS)
        assert set(json.loads(line)) == {
            "schema_version",
            "sensor_id",
            "station",
            "session_id",
            "sequence",
            "observation_time",
            "sent_at",
            "measurements",
        }


@pytest.mark.parametrize(
    "problem",
    [
        "mapping",
        "label",
        "infinity",
        "naive_time",
        "offset",
        "nonhour",
        "zero_sequence",
        "bool_sequence",
        "text_value",
        "missing_measurement",
        "version",
        "bad_uuid",
        "epoch_time",
        "epoch_text",
        "missing_version",
        "extra_envelope",
    ],
)
def test_strict_telemetry_contract(dataset, clock, sink, problem):
    PlaybackEngine(dataset, sink, speed=0, clock=clock).run()
    message = sink.messages[0].model_dump(mode="json")
    if problem == "mapping":
        message["station"] = "OK_Goodwell_2_SE"
    elif problem == "label":
        message["measurements"]["is_attacked"] = True
    elif problem == "infinity":
        message["measurements"]["P_CALC"] = float("inf")
    elif problem == "naive_time":
        message["observation_time"] = "2025-01-01T01:00:00"
    elif problem == "offset":
        message["sent_at"] = "2026-10-23T01:00:00+01:00"
    elif problem == "nonhour":
        message["observation_time"] = "2025-01-01T01:30:00Z"
    elif problem == "zero_sequence":
        message["sequence"] = 0
    elif problem == "bool_sequence":
        message["sequence"] = True
    elif problem == "text_value":
        message["measurements"]["P_CALC"] = "1.0"
    elif problem == "missing_measurement":
        del message["measurements"]["P_CALC"]
    elif problem == "version":
        message["schema_version"] = "2.0"
    elif problem == "epoch_time":
        message["observation_time"] = 1735693200
    elif problem == "epoch_text":
        message["observation_time"] = "1735693200"
    elif problem == "missing_version":
        del message["schema_version"]
    elif problem == "extra_envelope":
        message["is_attacked"] = False
    else:
        message["session_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        TelemetryMessage.model_validate_json(json.dumps(message))


def test_terminal_dispatch(dataset, clock, sink):
    engine = PlaybackEngine(dataset, sink, speed=0, clock=clock)
    assert dispatch_control("status", engine)["state"] == "ready"
    with pytest.raises(ValueError):
        dispatch_control("invalid", engine)
    assert dispatch_control("stop", engine)["state"] == "stopped"
    assert engine.run()["transmitted"] == 0


def test_sensor_selection_and_no_sequences_lost_on_last_pause(dataset, clock, sink):
    selected = dataset.select(["AZ_Tucson_11_W"])
    engine = PlaybackEngine(selected, sink, speed=0, clock=clock)
    sink.on_write = lambda message: engine.pause() if message.sequence == 3 else None
    status = engine.run()
    assert status["state"] == "completed"
    assert status["transmitted"] == 3
    for sensor in status["sensors"]:
        if sensor["station"] == "AZ_Tucson_11_W":
            assert sensor["selected"] and sensor["transmitted"] == 3
        else:
            assert sensor["state"] == "inactive" and sensor["transmitted"] == 0


def test_sensor_rejects_foreign_observations_and_nonincreasing_time(dataset, clock, sink):
    engine = PlaybackEngine(dataset, sink, speed=0, clock=clock)
    sensor = engine.sensors["AZ_Tucson_11_W"]
    observations = list(dataset.observations())
    with pytest.raises(ValueError, match="another sensor"):
        sensor.prepare(observations[1], clock.now())
    message = sensor.prepare(observations[0], clock.now())
    sensor.commit(message)
    with pytest.raises(ValueError, match="out-of-sequence"):
        sensor.commit(message)
    with pytest.raises(ValueError, match="strictly increase"):
        sensor.prepare(observations[0], clock.now())
