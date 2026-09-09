"""
Integration tests for the local web application server (app.py).
"""

import sys
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import ThreadedHTTPServer, SupportAppHandler


@pytest.fixture(scope="module")
def live_server():
    """Start the test server on an ephemeral free port."""
    server = ThreadedHTTPServer(("127.0.0.1", 0), SupportAppHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


def test_server_root_serves_html(live_server):
    """Test that GET / returns the frontend HTML."""
    req = urllib.request.Request(f"{live_server}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "AppleSupport AI Agent" in content
        assert "Live Simulator" in content


def test_api_health(live_server):
    """Test that GET /api/health returns healthy status."""
    req = urllib.request.Request(f"{live_server}/api/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "healthy"
        assert data["brand"] == "AppleSupport"


def test_api_metrics(live_server):
    """Test that GET /api/metrics returns metric summaries."""
    req = urllib.request.Request(f"{live_server}/api/metrics")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "metrics" in data
        assert "available_figures" in data
        assert len(data["available_figures"]) == 4


def test_api_taxonomy(live_server):
    """Test that GET /api/taxonomy returns discovered intents."""
    req = urllib.request.Request(f"{live_server}/api/taxonomy")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "intents" in data
        assert len(data["intents"]) >= 8


def test_figures_serving(live_server):
    """Test that figures are served with proper image MIME type."""
    req = urllib.request.Request(f"{live_server}/results/figures/baseline_comparison.png")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/png"
        content = resp.read()
        assert len(content) > 1000  # Valid PNG data


def test_chat_endpoint_validation(live_server):
    """Test POST /api/chat with empty message returns 400."""
    payload = json.dumps({"message": ""}).encode("utf-8")
    req = urllib.request.Request(
        f"{live_server}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
