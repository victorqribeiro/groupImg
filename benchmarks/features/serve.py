"""Serve the judging pages and save votes.

  python -I benchmarks/features/serve.py [--port 8765]

Open http://127.0.0.1:8765/round1/ . Votes are written to
results/roundN.json after every change. Listens on 127.0.0.1 only.
"""
import os
import re
import json
import argparse
import functools
import http.server

here = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):

    results = os.path.join(here, "results")

    def do_POST(self):
        m = re.fullmatch(r"/save/(\d{1,3})", self.path)
        if not m:
            return self.send_error(404)
        # another web page open in the browser can't send JSON here without
        # a CORS preflight, which this server never approves
        if self.headers.get("Content-Type") != "application/json":
            return self.send_error(415)
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send_error(403)
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= 1_000_000:
            return self.send_error(413)
        try:
            votes = json.loads(self.rfile.read(size))
        except ValueError:
            return self.send_error(400)
        os.makedirs(self.results, exist_ok=True)
        path = os.path.join(self.results, f"round{int(m[1])}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(votes, f, indent=1)
        os.replace(path + ".tmp", path)
        self.send_response(204)
        self.end_headers()

    def log_request(self, code="-", size="-"):
        # only saves and failures; image requests would flood the terminal
        if self.command == "POST" or not str(int(code)).startswith(("2", "3")):
            super().log_request(code, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default=os.path.join(here, "out"), help="pages to serve")
    ap.add_argument("--results", default=Handler.results, help="where votes go")
    args = ap.parse_args()
    Handler.results = args.results
    handler = functools.partial(Handler, directory=args.dir)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"http://127.0.0.1:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
