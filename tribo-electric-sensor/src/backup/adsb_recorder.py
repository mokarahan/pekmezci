#!/usr/bin/env python3
"""
ADS-B Exchange -> 10 s windows (50% overlap) -> CSV files.

Polls the ADS-B Exchange v2 "aircraft within radius" endpoint, keeps every
snapshot with its receive time, and writes one CSV per 10 s window, starting
a new window every 5 s. The filename carries the window's start time, in the
same local-time format as the microphone script, so the two can be matched.

CSV layout: one row per aircraft per poll.
    t              epoch seconds (UTC) when the poll completed
    server_now_ms  the API's "now" field (ms since epoch), if present
    hex, flight, r, t_type, lat, lon, ...   aircraft fields (see FIELDS)
Fields the API omits for an aircraft are left empty.

Install:  pip install requests
API key:  export ADSBX_API_KEY=...   (or pass --api-key)
Run:      python adsb_recorder.py --lat 41.0 --lon 29.0 --dist 100
Stop:     Ctrl+C  (an incomplete window is not written)
"""

import argparse
import csv
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_URL = "https://adsbexchange.com/api/aircraft/v2/lat/{lat}/lon/{lon}/dist/{dist}/"

WINDOW_SEC = 10.0
OVERLAP = 0.5
HOP_SEC = WINDOW_SEC * (1 - OVERLAP)      # 5.0

# (CSV column, key in the API's aircraft object). The API's "t" is the aircraft
# type, renamed here to "type_code" so it doesn't clash with the time column "t".
FIELDS = [
    ("hex", "hex"),
    ("flight", "flight"),
    ("registration", "r"),
    ("type_code", "t"),
    ("source", "type"),
    ("lat", "lat"),
    ("lon", "lon"),
    ("alt_baro", "alt_baro"),
    ("alt_geom", "alt_geom"),
    ("gs", "gs"),
    ("track", "track"),
    ("baro_rate", "baro_rate"),
    ("squawk", "squawk"),
    ("category", "category"),
    ("seen", "seen"),
    ("seen_pos", "seen_pos"),
    ("rssi", "rssi"),
    ("messages", "messages"),
]
COLUMNS = ["t", "server_now_ms"] + [c for c, _ in FIELDS]


def fetch_snapshot(session: requests.Session, url: str, headers: dict,
                   timeout: float) -> dict:
    t_start = time.time()
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    t_end = time.time()
    return {
        "t": (t_start + t_end) / 2,        # midpoint of the request
        "server_now_ms": data.get("now"),
        "ac": data.get("ac") or [],
    }


def write_window(out_dir: Path, start: float, snapshots: deque) -> Path | None:
    end = start + WINDOW_SEC
    selected = [s for s in snapshots if start <= s["t"] < end]
    if not selected:
        print(f"[warn] no data for window starting {start:.0f}; file skipped")
        return None

    dt = datetime.fromtimestamp(start)     # local time, matches mic script
    name = f"adsb_{dt:%Y%m%dT%H%M%S}.{dt.microsecond // 1000:03d}.csv"
    path = out_dir / name
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for snap in selected:
            for ac in snap["ac"]:
                row = [f"{snap['t']:.3f}", snap["server_now_ms"] or ""]
                for _, key in FIELDS:
                    v = ac.get(key, "")
                    row.append(v.strip() if isinstance(v, str) else v)
                w.writerow(row)
    os.replace(tmp, path)                  # atomic: readers never see half a file
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="ADS-B Exchange windowed CSV recorder")
    p.add_argument("--lat", type=float, required=True, help="centre latitude")
    p.add_argument("--lon", type=float, required=True, help="centre longitude")
    p.add_argument("--dist", type=float, default=100, help="radius in nautical miles")
    p.add_argument("--poll", type=float, default=1.0, help="seconds between API calls")
    p.add_argument("--out", default="adsb_data", help="output directory")
    p.add_argument("--api-key", default=os.environ.get("ADSBX_API_KEY"),
                   help="API key (default: $ADSBX_API_KEY)")
    p.add_argument("--url", default=DEFAULT_URL,
                   help="URL template with {lat} {lon} {dist} placeholders")
    p.add_argument("--header", default="api-auth",
                   help="name of the auth header (default: api-auth)")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout, seconds")
    args = p.parse_args()

    if not args.api_key:
        p.error("no API key: set ADSBX_API_KEY or pass --api-key")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.url.format(lat=args.lat, lon=args.lon, dist=args.dist)
    headers = {"Accept": "application/json", args.header: args.api_key}

    snapshots: deque = deque()
    # First window starts on the next multiple of the hop, so timestamps are tidy.
    next_start = math.ceil(time.time() / HOP_SEC) * HOP_SEC
    next_poll = time.monotonic()

    print(f"Polling every {args.poll}s; {WINDOW_SEC:.0f}s windows every {HOP_SEC:.0f}s. "
          f"First window starts at {datetime.fromtimestamp(next_start):%H:%M:%S}. "
          "Ctrl+C to stop.")

    try:
        with requests.Session() as session:
            while True:
                try:
                    snapshots.append(fetch_snapshot(session, url, headers, args.timeout))
                except (requests.RequestException, ValueError) as e:
                    print(f"[warn] poll failed: {e}")

                # Write every window that has fully elapsed.
                now = time.time()
                while now >= next_start + WINDOW_SEC:
                    path = write_window(out_dir, next_start, snapshots)
                    if path:
                        print(f"saved {path.name}")
                    next_start += HOP_SEC
                    while snapshots and snapshots[0]["t"] < next_start:
                        snapshots.popleft()

                next_poll += args.poll
                time.sleep(max(0.0, next_poll - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
