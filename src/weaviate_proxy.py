import os
import http.server
import socketserver
import requests

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WEAVIATE_URL or not API_KEY:
    raise RuntimeError("WEAVIATE_URL e WEAVIATE_API_KEY devono essere impostate nel terminale.")

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        resp = requests.post(
            WEAVIATE_URL + self.path,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": self.headers.get("Content-Type", "application/json"),
            },
            data=body,
        )

        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() not in ["content-length", "transfer-encoding", "connection"]:
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp.content)

    def do_GET(self):
        resp = requests.get(
            WEAVIATE_URL + self.path,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() not in ["content-length", "transfer-encoding", "connection"]:
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp.content)


if __name__ == "__main__":
    PORT = 9000
    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        print(f"Proxy Weaviate attivo su http://localhost:{PORT}")
        httpd.serve_forever()
