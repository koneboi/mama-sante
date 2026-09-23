from __future__ import annotations

_TRAVEL_TIME_WEIGHT = "travel_time"  # seconds, added by scripts/fetch_network.py
_DETOUR_FACTOR = 1.3                # haversine -> road-distance factor for remote fallback
_DEFAULT_SPEED_KMH = 40.0
_TRANSPORT_PER_KM = 150.0           # FCFA/km, added to out-of-pocket cost for long trips


def travel_metrics(origin: dict, dest: dict, G=None) -> tuple[float, float]:
    """Return (distance_km, travel_h) between two lat/lon points.

    Uses the OSM road network when G is provided and covers both points;
    otherwise estimates via haversine * detour factor.
    """
    straight_km = _haversine(origin, dest)
    fallback_h = straight_km * _DETOUR_FACTOR / _DEFAULT_SPEED_KMH

    if G is None:
        return straight_km * _DETOUR_FACTOR, fallback_h

    try:
        import networkx as nx
        import osmnx as ox

        u = ox.nearest_nodes(G, X=origin["lon"], Y=origin["lat"])
        v = ox.nearest_nodes(G, X=dest["lon"], Y=dest["lat"])
        # only trust the network when both points project to a nearby node (<2 km) —
        # otherwise snap noise from off-graph locations would corrupt travel time
        if G.has_node(u) and G.has_node(v):
            off = (
                _haversine(dest, {"lat": G.nodes[v]["y"], "lon": G.nodes[v]["x"]}) > 2.0
                or _haversine(origin, {"lat": G.nodes[u]["y"], "lon": G.nodes[u]["x"]}) > 2.0
            )
            if not off:
                sample = next(iter(G.edges(data=True)))[2] if G.number_of_edges() else {}
                if _TRAVEL_TIME_WEIGHT in sample:
                    travel_h = nx.shortest_path_length(G, u, v, weight=_TRAVEL_TIME_WEIGHT) / 3600.0
                else:
                    dist_m = nx.shortest_path_length(G, u, v, weight="length")
                    travel_h = dist_m / 1000.0 / _DEFAULT_SPEED_KMH
                return straight_km * _DETOUR_FACTOR, travel_h
    except (nx.NetworkXNoPath, StopIteration, KeyError):
        pass

    return straight_km * _DETOUR_FACTOR, fallback_h


def min_cost_center(origin: dict, centers: list[dict], weights: dict, G=None, max_travel_ratio: float | None = 1.5) -> dict:
    """Pick the center minimizing weighted travel/cost/load among capable candidates.

    Time-first: when max_travel_ratio is set, candidates are first narrowed to those
    within `max_travel_ratio * (nearest candidate travel)` so a cheap-but-far center
    never outranks the nearest suitable one.
    """
    scored = []
    for c in centers:
        dist_km, travel_h = travel_metrics(origin, c, G)
        out_of_pocket = c.get("out_of_pocket", 0.0) + dist_km * _TRANSPORT_PER_KM
        cost = (
            weights["travel_km"] * dist_km
            + weights["travel_h"] * travel_h
            + weights["out_of_pocket"] * out_of_pocket
            + weights["center_load"] * c.get("load", 0.0)
        )
        cd = dict(c)
        cd["distance_km"] = dist_km
        cd["travel_h"] = travel_h
        cd["out_of_pocket"] = out_of_pocket
        cd["_cost"] = cost
        scored.append(cd)

    if scored and max_travel_ratio:
        tmin = min(cd["travel_h"] for cd in scored)
        if tmin == 0:
            tmin = 1e-9
        budget = tmin * max_travel_ratio
        scored = [cd for cd in scored if cd["travel_h"] <= budget]

    best, best_cost = None, float("inf")
    for cd in scored:
        if cd["_cost"] < best_cost:
            best, best_cost = cd, cd["_cost"]
    return best


def _haversine(a: dict, b: dict) -> float:
    from math import asin, cos, radians, sin, sqrt

    r = 6371.0
    dlat = radians(b["lat"] - a["lat"])
    dlon = radians(b["lon"] - a["lon"])
    h = sin(dlat / 2) ** 2 + cos(radians(a["lat"])) * cos(radians(b["lat"])) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(h))