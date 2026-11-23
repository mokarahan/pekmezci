#!/usr/bin/env python3
"""
adsb.lol -> per-flight tracks -> 1 Hz cubic-spline interpolation -> one CSV per second.

You give a rectangular area (lat/lon min/max). adsb.lol's public API only
supports circle queries, so the script asks for the smallest circle centred on the
rectangle that covers all four corners, then discards aircraft outside the rectangle.
The covering circle must fit the API's radius limit (250 nm by default).

API key: taken from --api-key or the ADSBLOL_API_KEY environment variable and sent
in the header named by --key-header (default "api-auth"). If no key is given, the
free keyless request is used. If a key is given but rejected (HTTP 401/403), that
poll is retried without a key.

Raw data: the file of the FIRST SECOND of every segment is not interpolated; it
holds what was received from the service in that poll, with the same values the
API sent (every field, nothing stripped or rounded) for the aircraft inside your
rectangle only; flights outside it are left out.
So each poll's second gets a raw file, and the seconds between two polls get
interpolated files. The raw file has one row per aircraft; its columns are
    poll_time      epoch seconds (UTC) when the poll completed
    api_<key>      the response's own top-level values (now, total, ctime, ...)
    <field>...     every aircraft field under the API's name; a field an aircraft
                   lacks is empty; list/object fields are written as JSON
It is written when the next poll arrives (and on Ctrl+C for the last poll), so it
appears one poll interval after the reading. --no-raw turns the raw files off
(then every second is interpolated).

Tracks: for every aircraft that has a "flight" (callsign) the script keeps its recent
(time, lat, lon, altitude) points. A track is deleted when no message has been
received from that aircraft for 5 minutes (based on the API's "seen" field).

Polling: every 100 s by default. After each poll, every aircraft that got a new
position is interpolated with a cubic spline (using its recent points) at every whole
second between its previous position and its newest one. Each second is written to
its OWN file (the poll's own second is the raw file described above), adsb_YYYYmmddTHHMMSS.000.csv (local time, same format as the
microphone script). Segments never overlap: each second is written once per aircraft.
Output is therefore delayed by one poll interval, because interpolation needs the
newest point as its end.

CSV layout: one row per aircraft per second.
    t        epoch seconds (UTC), whole second
    flight, hex, lat, lon, alt     alt in feet (alt_baro, else alt_geom; "ground" = 0)

Install:  pip install requests numpy scipy
Run:      python adsb_lol_recorder.py --lat-min 40.5 --lat-max 41.5 --lon-min 28.5 --lon-max 29.5
Note:     the rectangle must not cross the antimeridian (lon-min < lon-max).
Stop:     Ctrl+C
"""

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from scipy.interpolate import CubicSpline

DEFAULT_URL = "https://api.adsb.lol/v2/point/{lat}/{lon}/{dist}"

DEFAULT_POLL_SEC = 100.0
#TRACK_TIMEOUT_SEC = 300.0                 # drop a flight after 5 min of silence
TRACK_TIMEOUT_SEC = 3*DEFAULT_POLL_SEC                 # drop a flight after 3*DEFAULT_POLL_SEC seconds of silence

MAX_POINTS = 8                            # points kept per flight for the spline

EARTH_RADIUS_NM = 3440.065
MAX_RADIUS_NM = 250.0                     # adsb.lol radius limit, as far as I know

COLUMNS = ["t", "flight", "hex", "lat", "lon", "alt"]


# --------------------------------------------------------------------------- geometry

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(a))


def covering_circle(lat_min, lat_max, lon_min, lon_max):
    """Centre of the rectangle and the radius (nm) reaching its farthest corner."""
    lat_c = (lat_min + lat_max) / 2
    lon_c = (lon_min + lon_max) / 2
    radius = max(haversine_nm(lat_c, lon_c, la, lo)
                 for la in (lat_min, lat_max) for lo in (lon_min, lon_max))
    return lat_c, lon_c, math.ceil(radius) + 1     # +1 nm safety margin


def in_box(ac: dict, box: tuple) -> bool:
    lat_min, lat_max, lon_min, lon_max = box
    lat, lon = ac.get("lat"), ac.get("lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return False                       # no position -> cannot be inside the box
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


# --------------------------------------------------------------------------- fetching

def fetch_snapshot(session: requests.Session, url: str, headers: dict,
                   api_key: str | None, key_header: str,
                   timeout: float, box: tuple) -> dict:
    t_start = time.time()
    resp = None
    if api_key:
        resp = session.get(url, headers={**headers, key_header: api_key}, timeout=timeout)
        if resp.status_code in (401, 403):
            print(f"[warn] API key rejected (HTTP {resp.status_code}); retrying without key")
            resp = None
    if resp is None:
        resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    t_end = time.time()
    return {
        "t": (t_start + t_end) / 2,        # midpoint of the request
        "data": data,                      # full response, for the raw CSV
        "ac": [ac for ac in (data.get("ac") or []) if in_box(ac, box)],
    }


# --------------------------------------------------------------------------- tracks

def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def altitude_ft(ac: dict) -> float:
    alt = ac.get("alt_baro")
    if alt == "ground":
        return 0.0
    alt = _num(alt)
    if alt is None:
        alt = _num(ac.get("alt_geom"))
    return float("nan") if alt is None else alt


class Track:
    __slots__ = ("hex", "last_seen", "pts", "emitted_until")

    def __init__(self, hex_code: str):
        self.hex = hex_code
        self.last_seen = 0.0               # epoch s of the last message of any kind
        self.pts = deque(maxlen=MAX_POINTS)  # (t_obs, lat, lon, alt)
        self.emitted_until = None          # last whole second already written


def update_tracks(tracks: dict, snapshot: dict) -> None:
    t_poll = snapshot["t"]
    for ac in snapshot["ac"]:
        flight = (ac.get("flight") or "").strip()
        if not flight:
            continue                       # records are keyed by flight
        tr = tracks.get(flight)
        if tr is None:
            tr = tracks[flight] = Track(ac.get("hex", ""))
        tr.hex = ac.get("hex", tr.hex)

        seen = _num(ac.get("seen")) or 0.0
        tr.last_seen = max(tr.last_seen, t_poll - seen)

        # Time the position was actually received (the API reports its age).
        pos_age = _num(ac.get("seen_pos"))
        t_obs = t_poll - (seen if pos_age is None else pos_age)
        if tr.pts and t_obs <= tr.pts[-1][0] + 0.5:
            continue                       # same position as before, nothing new
        tr.pts.append((t_obs, float(ac["lat"]), float(ac["lon"]), altitude_ft(ac)))


def purge_tracks(tracks: dict, now: float) -> int:
    stale = [f for f, tr in tracks.items() if now - tr.last_seen > TRACK_TIMEOUT_SEC]
    for f in stale:
        del tracks[f]
    return len(stale)


# --------------------------------------------------------------------------- interpolation

def spline_eval(ts: np.ndarray, vs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(vs)
    ts, vs = ts[ok], vs[ok]
    if len(ts) < 2:
        return np.full(len(grid), np.nan)
    if len(ts) == 2:                       # seen only twice -> plain linear interpolation
        return np.interp(grid, ts, vs, left=np.nan, right=np.nan)
    # 3 points -> parabola, 4+ -> full cubic spline (not-a-knot).
    return CubicSpline(ts, vs, extrapolate=False)(grid)


def interpolate_new_segment(flight: str, tr: Track) -> dict:
    """Rows for every whole second between the previous and the newest point."""
    if len(tr.pts) < 2:
        return {}
    t_prev, t_last = tr.pts[-2][0], tr.pts[-1][0]
    first = tr.emitted_until + 1 if tr.emitted_until is not None else math.ceil(t_prev)
    last = math.floor(t_last)
    if last < first:
        return {}

    arr = np.array(tr.pts, dtype=float)    # columns: t, lat, lon, alt
    grid = np.arange(first, last + 1, dtype=float)
    lat = spline_eval(arr[:, 0], arr[:, 1], grid)
    lon = spline_eval(arr[:, 0], arr[:, 2], grid)
    alt = spline_eval(arr[:, 0], arr[:, 3], grid)

    def fmt(v, nd):
        return "" if np.isnan(v) else f"{v:.{nd}f}"

    rows = defaultdict(list)
    for s, la, lo, al in zip(grid, lat, lon, alt):
        rows[int(s)].append([int(s), flight, tr.hex, fmt(la, 6), fmt(lo, 6), fmt(al, 1)])
    tr.emitted_until = last
    return rows


# --------------------------------------------------------------------------- output

def write_second_file(out_dir: Path, second: int, rows: list) -> Path:
    dt = datetime.fromtimestamp(second)    # local time, matches mic script
    path = out_dir / f"adsb_{dt:%Y%m%dT%H%M%S}.{dt.microsecond // 1000:03d}.csv"
    if path.exists():                      # another flight already wrote this second
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        return path
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    os.replace(tmp, path)                  # atomic: readers never see half a file
    return path


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return v


def write_raw_csv(out_dir: Path, t_poll: float, data: dict) -> Path:
    """Save the whole API response (every aircraft, every field) as the file of
    the poll's own second, named like the interpolated per-second files."""
    aircraft = data.get("ac") or []
    api_keys = sorted(k for k in data if k != "ac")
    ac_keys = sorted({k for ac in aircraft for k in ac},
                     key=lambda k: (k not in ("hex", "flight"), k))   # hex, flight first

    dt = datetime.fromtimestamp(math.floor(t_poll))
    path = out_dir / f"adsb_{dt:%Y%m%dT%H%M%S}.000.csv"
    n = 2
    while path.exists():                   # two polls in one second: never overwrite
        path = out_dir / f"adsb_{dt:%Y%m%dT%H%M%S}.000_{n}.csv"
        n += 1
    tmp = path.with_suffix(".tmp")
    prefix = [f"{t_poll:.3f}"] + [_cell(data[k]) for k in api_keys]
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["poll_time"] + [f"api_{k}" for k in api_keys] + ac_keys)
        for ac in aircraft:
            w.writerow(prefix + [_cell(ac.get(k)) for k in ac_keys])
    os.replace(tmp, path)                  # atomic: readers never see half a file
    return path


# --------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="adsb.lol 1 Hz interpolating CSV recorder")
    p.add_argument("--lat-min", type=float, required=True, help="southern edge, degrees")
    p.add_argument("--lat-max", type=float, required=True, help="northern edge, degrees")
    p.add_argument("--lon-min", type=float, required=True, help="western edge, degrees")
    p.add_argument("--lon-max", type=float, required=True, help="eastern edge, degrees")
    p.add_argument("--max-radius", type=float, default=MAX_RADIUS_NM,
                   help="largest circle radius the API accepts, nm (default 250)")
    p.add_argument("--poll", type=float, default=DEFAULT_POLL_SEC,
                   help="seconds between API calls (default 100)")
    p.add_argument("--api-key", default=os.environ.get("ADSBLOL_API_KEY"),
                   help="API key (default: $ADSBLOL_API_KEY); keyless if absent")
    #p.add_argument("--key-header", default="api-auth",
    p.add_argument("--key-header", default="x-api-key",
                   help="HTTP header that carries the key (default api-auth)")
    p.add_argument("--no-raw", action="store_true",
                   help="no raw first-second files; interpolate every second")
    p.add_argument("--out", default="adsb_data", help="output directory")
    p.add_argument("--url", default=DEFAULT_URL,
                   help="URL template with {lat} {lon} {dist} placeholders")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout, seconds")
    args = p.parse_args()

    if not (-90 <= args.lat_min < args.lat_max <= 90):
        p.error("need -90 <= lat-min < lat-max <= 90")
    if not (-180 <= args.lon_min < args.lon_max <= 180):
        p.error("need -180 <= lon-min < lon-max <= 180 (no antimeridian crossing)")

    box = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lat_c, lon_c, radius = covering_circle(*box)
    if radius > args.max_radius:
        p.error(f"rectangle needs a {radius} nm covering circle, above the "
                f"{args.max_radius:.0f} nm limit; use a smaller rectangle")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.url.format(lat=f"{lat_c:.5f}", lon=f"{lon_c:.5f}", dist=radius)
    headers = {"Accept": "application/json", "User-Agent": "adsb-interp-recorder/1.0"}

    tracks: dict[str, Track] = {}
    pending = None                         # (poll time, full response) not yet written raw
    raw_seconds: set[int] = set()          # whole seconds that hold a raw file
    next_poll = time.monotonic()

    print(f"Rectangle lat {args.lat_min}..{args.lat_max}, lon {args.lon_min}..{args.lon_max} "
          f"-> querying circle at ({lat_c:.4f}, {lon_c:.4f}), {radius} nm, then filtering.")
    print(f"Polling every {args.poll:g}s "
          f"({'with API key' if args.api_key else 'no API key'}); "
          f"1 s cubic-spline records, one file per second. Ctrl+C to stop.")

    try:
        with requests.Session() as session:
            while True:
                try:
                    snap = fetch_snapshot(session, url, headers, args.api_key,
                                          args.key_header, args.timeout, box)
                    if not args.no_raw:
                        raw_seconds.add(math.floor(snap["t"]))
                        raw_seconds = {s for s in raw_seconds if s > snap["t"] - 1200}
                        if pending is not None:    # raw file of the previous poll's second
                            write_raw_csv(out_dir, *pending)
                        # keep only the aircraft inside the rectangle (snap["ac"] is already filtered)
                        pending = (snap["t"], {**snap["data"], "ac": snap["ac"]})
                    update_tracks(tracks, snap)

                    per_second = defaultdict(list)
                    for flight, tr in tracks.items():
                        for sec, rows in interpolate_new_segment(flight, tr).items():
                            per_second[sec].extend(rows)
                    for sec in sorted(per_second):
                        if sec in raw_seconds:     # that second is the raw file
                            del per_second[sec]
                    for sec in sorted(per_second):
                        write_second_file(out_dir, sec, per_second[sec])
                    print(f"poll ok: {len(snap['ac'])} aircraft in box, "
                          f"{len(tracks)} tracked, {len(per_second)} second-files written")
                except (requests.RequestException, ValueError) as e:
                    print(f"[warn] poll failed: {e}")

                dropped = purge_tracks(tracks, time.time())
                if dropped:
                    print(f"purged {dropped} flight(s) silent for > {TRACK_TIMEOUT_SEC:.0f}s")

                next_poll += args.poll
                time.sleep(max(0.0, next_poll - time.monotonic()))
    except KeyboardInterrupt:
        if pending is not None:            # raw file of the last poll
            write_raw_csv(out_dir, *pending)
        print("\nStopped.")


if __name__ == "__main__":
    main()
