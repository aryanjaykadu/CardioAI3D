"""
Serve the CardioAI dashboard locally (no build step, no internet needed).

    python start_dashboard.py               # http://localhost:8080, opens your browser
    python start_dashboard.py --port 9000 --no-browser
"""

import argparse
import http.server
import os
import socketserver
import webbrowser

DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    for port in range(args.port, args.port + 20):
        try:
            httpd = socketserver.TCPServer(("", port), Handler)
        except OSError:
            continue
        url = f"http://localhost:{port}/"
        print(f"CardioAI dashboard: {url}  (Ctrl+C to stop)", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return
    raise SystemExit(f"No free port in {args.port}-{args.port + 19}")


if __name__ == "__main__":
    main()
