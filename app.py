"""
Production Web Application Server for AppleSupport AI Support Agent.

Zero-dependency HTTP server delivering an interactive web dashboard:
- Live Agent Simulator & Pipeline Tracer
- 3-Way Model Arena (Agent vs Simple vs Trivial)
- Empirical Benchmark Metrics & Visual Gallery
- Intent Taxonomy & Knowledge Base Explorer

Usage:
    python app.py
    python app.py --port 8080
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse
import mimetypes

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BRAND_NAME, RESULTS_DIR, INTENT_TAXONOMY_PATH, MODEL_CLASSIFY, MODEL_GENERATE
)
from agent.pipeline import SupportAgent
from baselines.trivial import TrivialBaseline
from baselines.simple import SimpleBaseline

# Global singletons
_agent = None
_trivial = None
_simple = None


def get_models():
    """Lazily instantiate models as singletons."""
    global _agent, _trivial, _simple
    if _agent is None:
        print("[App] Initializing SupportAgent pipeline...")
        _agent = SupportAgent()
    if _trivial is None:
        print("[App] Initializing TrivialBaseline...")
        _trivial = TrivialBaseline()
        _trivial.fit()
    if _simple is None:
        print("[App] Initializing SimpleBaseline...")
        _simple = SimpleBaseline()
        _simple.fit()
    return _agent, _trivial, _simple


class SupportAppHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for the AI Support Agent Dashboard."""

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200, "text/plain")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Static root -> web/index.html
        if path in ("/", "/index.html"):
            index_path = PROJECT_ROOT / "web" / "index.html"
            if not index_path.exists():
                self._set_headers(404, "text/plain")
                self.wfile.write(b"web/index.html not found.")
                return

            self._set_headers(200, "text/html; charset=utf-8")
            with open(index_path, "rb") as f:
                self.wfile.write(f.read())
            return

        # Serve visual evaluation figures: /results/figures/<filename>
        if path.startswith("/results/figures/"):
            filename = os.path.basename(path)
            fig_path = RESULTS_DIR / "figures" / filename
            if fig_path.exists() and fig_path.is_file():
                mime, _ = mimetypes.guess_type(str(fig_path))
                self._set_headers(200, mime or "image/png")
                with open(fig_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._set_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": "Figure not found"}).encode("utf-8"))
                return

        # API: Health check
        if path == "/api/health":
            self._set_headers(200, "application/json")
            resp = {
                "status": "healthy",
                "brand": BRAND_NAME,
                "models": {
                    "classify": MODEL_CLASSIFY,
                    "generate": MODEL_GENERATE,
                }
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # API: Metrics summary and judge agreement
        if path == "/api/metrics":
            metrics_path = RESULTS_DIR / "metrics_summary.json"
            agreement_path = RESULTS_DIR / "judge_agreement.json"

            metrics_data = {}
            agreement_data = {}

            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)

            if agreement_path.exists():
                with open(agreement_path, "r", encoding="utf-8") as f:
                    agreement_data = json.load(f)

            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({
                "metrics": metrics_data,
                "judge_agreement": agreement_data,
                "available_figures": [
                    "baseline_comparison.png",
                    "llm_judge_dimensions.png",
                    "judge_agreement.png",
                    "intent_confusion_matrix.png"
                ]
            }).encode("utf-8"))
            return

        # API: Intent taxonomy
        if path == "/api/taxonomy":
            if INTENT_TAXONOMY_PATH.exists():
                with open(INTENT_TAXONOMY_PATH, "r", encoding="utf-8") as f:
                    taxonomy_data = json.load(f)
            else:
                taxonomy_data = {"error": "Taxonomy file not found"}

            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps(taxonomy_data).encode("utf-8"))
            return

        # Fallback 404
        self._set_headers(404, "application/json")
        self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Read JSON body
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            data = json.loads(post_body.decode("utf-8"))
        except Exception as e:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {str(e)}"}).encode("utf-8"))
            return

        message = data.get("message", "").strip()
        if not message:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": "Field 'message' cannot be empty"}).encode("utf-8"))
            return

        agent, trivial, simple = get_models()

        # Endpoint: POST /api/chat (single agent pipeline)
        if path == "/api/chat":
            t0 = time.time()
            try:
                result = agent.process_message(message)
                elapsed_ms = round((time.time() - t0) * 1000, 1)
                result["latency_ms"] = elapsed_ms

                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps({"success": True, "data": result}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": f"Pipeline failure: {str(e)}"}).encode("utf-8"))
            return

        # Endpoint: POST /api/compare (Agent vs Simple vs Trivial)
        if path == "/api/compare":
            try:
                # 1. Agent
                t_agent = time.time()
                agent_res = agent.process_message(message)
                agent_ms = round((time.time() - t_agent) * 1000, 1)
                agent_res["latency_ms"] = agent_ms

                # 2. Simple baseline
                t_simple = time.time()
                simple_res = simple.predict(message)
                simple_ms = round((time.time() - t_simple) * 1000, 1)
                simple_res["latency_ms"] = simple_ms

                # 3. Trivial baseline
                t_trivial = time.time()
                trivial_res = trivial.predict(message)
                trivial_ms = round((time.time() - t_trivial) * 1000, 1)
                trivial_res["latency_ms"] = trivial_ms

                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps({
                    "success": True,
                    "customer_message": message,
                    "agent": agent_res,
                    "simple": simple_res,
                    "trivial": trivial_res
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": f"Comparison failure: {str(e)}"}).encode("utf-8"))
            return

        # Fallback 404
        self._set_headers(404, "application/json")
        self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Custom clean log format
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} - {args[1]} {args[2]}\n")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for snappy concurrency."""
    daemon_threads = True
    allow_reuse_address = True


def run_server(host="127.0.0.1", port=8000):
    """Start the web server and print access instructions."""
    # Pre-warm models so first user click is instant
    print("\n" + "=" * 65)
    print(f"  Starting AppleSupport AI Support Agent Web Server")
    print("=" * 65)
    get_models()

    server = ThreadedHTTPServer((host, port), SupportAppHandler)
    url = f"http://{host}:{port}"
    print("\n" + "-" * 65)
    print(f"  [SUCCESS] Server running and ready for traffic!")
    print(f"  -> Local Dashboard URL: {url}")
    print(f"  -> Press Ctrl+C to stop the server")
    print("-" * 65 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AppleSupport AI Support Agent Web Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
