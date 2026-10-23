import io
import socket
import time
from threading import Thread

import pytest
import uvicorn

from sensorlab.dataset import DetectorDataset, parse_utc
from sensorlab.gate.api import create_app
from sensorlab.gate.http_transport import EnvelopeFanout, HTTPSink
from sensorlab.gate.tls import development_certificates
from sensorlab.playback import PlaybackEngine
from sensorlab.signing import SignedEnvelope, SignedTelemetrySink
from sensorlab.transport import NDJSONSink
from sensorlab.wire import SecurityError


@pytest.fixture(scope="module")
def tls_material(tmp_path_factory):
    directory = tmp_path_factory.mktemp("https-test") / "tls"
    password = b"synthetic-TLS-password-not-production"
    development_certificates(directory, password)
    return directory, password


@pytest.fixture
def live_gate(gate_environment, clock, tls_material):
    config, _, repository, message, signer = gate_environment
    directory, password = tls_material
    application = create_app(config, clock)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            application,
            host="127.0.0.1",
            port=port,
            ssl_certfile=str(directory / "server.pem"),
            ssl_keyfile=str(directory / "server.private.pem"),
            ssl_keyfile_password=password.decode(),
            proxy_headers=False,
            access_log=False,
            log_level="warning",
        )
    )
    thread = Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield (
            f"https://127.0.0.1:{port}/v1/telemetry",
            directory / "ca.pem",
            repository,
            message,
            signer,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()


def test_https_adapter_and_duplicate_rejection(live_gate):
    endpoint, ca, repository, message, signer = live_gate
    sink = HTTPSink(endpoint, ca_cert=ca)
    try:
        sink.write(signer.sign(message))
        assert sink.accepted == 1
        with pytest.raises(SecurityError, match="E_REPLAY_SEQ"):
            sink.write(signer.sign(message))
    finally:
        sink.close()
    assert repository.status()["accepted"] == 1
    assert repository.status()["rejected"] == 1


def test_certificate_verification_cannot_be_disabled(live_gate):
    endpoint, _, repository, message, signer = live_gate
    sink = HTTPSink(endpoint)  # Development CA is deliberately absent from system trust.
    try:
        with pytest.raises(SecurityError, match="HTTPS telemetry submission failed"):
            sink.write(signer.sign(message))
    finally:
        sink.close()
    assert repository.status()["accepted"] == 0


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/v1/telemetry",
        "https://u:p@localhost/v1/telemetry",
        "https://localhost/other",
        "https://localhost/v1/telemetry?x=1",
    ],
)
def test_unsafe_endpoint_rejected(url):
    with pytest.raises(ValueError):
        HTTPSink(url)


@pytest.mark.real_dataset
def test_real_combined_144_over_verified_https(real_release, live_gate, clock):
    endpoint, ca, repository, message, signer = live_gate
    selected = DetectorDataset.load(real_release).select(
        start=parse_utc("2025-01-01T01:00:00Z"), end=parse_utc("2025-01-02T01:00:00Z")
    )
    stream = io.StringIO()
    http = HTTPSink(endpoint, ca_cert=ca)
    try:
        sink = SignedTelemetrySink(signer, EnvelopeFanout(http, NDJSONSink(stream)))
        result = PlaybackEngine(
            selected, sink, speed=0, clock=clock, session_id=message.session_id
        ).run()
        assert http.accepted == result["transmitted"] == 144
    finally:
        http.close()
    status = repository.status()
    assert status["accepted"] == status["outbox_count"] == 144
    assert status["rejected"] == status["sequence_gap_events"] == 0
    assert all(row["last_sequence"] == 24 for row in status["sensors"])
    envelopes = [
        SignedEnvelope.model_validate_json(line) for line in stream.getvalue().splitlines()
    ]
    for envelope, observation in zip(envelopes, selected.observations(), strict=True):
        assert envelope.telemetry.measurements == observation.measurements
        assert envelope.telemetry.observation_time == observation.time
