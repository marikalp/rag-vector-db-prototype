import os
import http.server
import socketserver
import requests
import json

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WEAVIATE_URL or not API_KEY:
    raise RuntimeError("WEAVIATE_URL e WEAVIATE_API_KEY devono essere impostate nel terminale.")

class ProxyHandler(http.server.BaseHTTPRequestHandler):

    def _send_json(self, obj, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        # Risposte locali per readiness check del client v3
        if self.path == "/v1/.well-known/ready":
            return self._send_json({"status": "ready"})
        if self.path == "/v1/.well-known/live":
            return self._send_json({"status": "live"})

        # Inoltro GET a Weaviate Cloud
        target_url = WEAVIATE_URL + self.path
        headers = {"Authorization": f"Bearer {API_KEY}"}

        resp = requests.get(target_url, headers=headers)

        self.send_response(resp.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.content)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        target_url = WEAVIATE_URL + self.path
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        resp = requests.post(target_url, headers=headers, data=body)

        self.send_response(resp.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.content)

    def do_DELETE(self):
        target_url = WEAVIATE_URL + self.path
        headers = {"Authorization": f"Bearer {API_KEY}"}

        resp = requests.delete(target_url, headers=headers)

        self.send_response(resp.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.content)


if __name__ == "__main__":
    PORT = 9000
    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        print(f"Proxy Weaviate attivo su http://localhost:{PORT}")
        httpd.serve_forever()

