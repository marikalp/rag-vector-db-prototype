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
    def _forward(self, method):
        target_url = WEAVIATE_URL + self.path

        # Headers da inoltrare
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": self.headers.get("Content-Type", "application/json")
        }

        # Body (solo per POST/PUT)
        body = None
        if method in ["POST", "PUT"]:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

        # Inoltro
        resp = requests.request(method, target_url, headers=headers, data=body)

        # Risposta al client
        self.send_response(resp.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Se il body non è JSON valido, rispondi comunque con {}
        try:
            self.wfile.write(resp.content)
        except:
            self.wfile.write(b"{}")

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_PUT(self):
        self._forward("PUT")

    def do_DELETE(self):
        self._forward("DELETE")


if __name__ == "__main__":
    PORT = 9000
    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        print(f"Proxy Weaviate attivo su http://localhost:{PORT}")
        httpd.serve_forever()

