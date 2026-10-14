"""
Launch script for the 3D Web Clinical Decision Support Dashboard.
Hosts the dashboard on a local HTTP server and automatically opens it in the browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.join(os.path.dirname(__file__), "dashboard")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()


def run():
    global PORT
    while PORT < 8100:
        try:
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                url = f"http://localhost:{PORT}/index.html"
                print("=" * 70)
                print("CARDIOAI 3D - CLINICAL DECISION SUPPORT DASHBOARD")
                print("=" * 70)
                print(f"Serving dashboard at: {url}")
                print("Opening web browser...")
                webbrowser.open(url)
                print("\nServer running. Press Ctrl+C to terminate.")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nServer stopped.")
                    break
        except OSError:
            PORT += 1


if __name__ == "__main__":
    run()
