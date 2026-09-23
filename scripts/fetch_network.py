"""Add realistic per-edge travel times to a street/road network and save as GraphML."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GEO_DIR = os.path.join(ROOT, "data", "geo")

# rough driving speed (km/h) by OSM highway class
SPEED_KMH = {
    "motorway": 100,
    "trunk": 80,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "unclassified": 30,
    "residential": 25,
    "living_street": 15,
    "service": 20,
    "track": 15,
}


def add_travel_time(G) -> None:
    for u, v, k, d in G.edges(keys=True, data=True):
        highway = d.get("highway")
        speed = d.get("maxspeed")
        if isinstance(highway, list):
            highway = highway[0]
        if speed:
            try:
                speed_kmh = float(str(speed).replace(" km/h", "").replace("mph", "").strip())
            except ValueError:
                speed_kmh = SPEED_KMH.get(highway, 30)
        else:
            speed_kmh = SPEED_KMH.get(highway, 30)
        length_m = d["length"]
        d["travel_time"] = length_m / 1000.0 / speed_kmh * 3600.0  # seconds


def fetch(place: str, network_type: str = "drive", out: str = "") -> str:
    import osmnx as ox

    G = ox.graph_from_place(place, network_type=network_type)
    add_travel_time(G)
    out = out or os.path.join(GEO_DIR, place.replace(" ", "_").replace(",", "").lower() + ".graphml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ox.save_graphml(G, out)
    print(f"saved {len(G.nodes)} nodes / {len(G.edges)} edges -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Bamako, Mali")
    p.add_argument("--network-type", default="drive")
    p.add_argument("--output", default="")
    a = p.parse_args()
    fetch(a.place, a.network_type, a.output)