"""Local dashboard comparing benchmark results of all models.

    python -m evaluation.dashboard                 # http://localhost:8000
    python -m evaluation.dashboard --port 8080 --host 0.0.0.0

Serves ``index.html`` and the results CSV (``evaluation/results/benchmark_v1.csv``,
written by ``python -m evaluation.export_results``). Needs only the Python standard
library, so it runs on the host without Docker. Over SSH, either forward the port
(``ssh -L 8000:localhost:8000 <host>``) or listen on all interfaces with ``--host 0.0.0.0``.

The page can also be opened without the server: open ``index.html`` in a browser and
load a CSV with the "Load CSV" button.
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "results" / "benchmark_v1.csv"


class Handler(http.server.SimpleHTTPRequestHandler):
    csv_path: Path = DEFAULT_CSV

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?")[0] == "/results.csv":
            if not self.csv_path.is_file():
                self.send_error(404, f"{self.csv_path} not found; run "
                                     "python -m evaluation.export_results")
                return
            body = self.csv_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        super().do_GET()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.dashboard",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    Handler.csv_path = args.csv.resolve()
    handler = functools.partial(Handler, directory=str(HERE))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Dashboard: http://{'localhost' if args.host == '127.0.0.1' else args.host}:"
              f"{args.port}  (results: {Handler.csv_path})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
