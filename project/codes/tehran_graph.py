# -*- coding: utf-8 -*-
"""
Tehran road graph builder.

Reads nodes.csv and edges.csv from the script's directory, fills the empty
distance_km column with the Haversine distance between node coordinates,
builds a directed NetworkX graph, plots it with a geographic layout
(lon = x, lat = y), and finds a shortest route (Dijkstra, weight = distance_km).

Outputs (written next to the script):
    - edges_with_distance.csv  (utf-8-sig, distance_km filled)
    - tehran_graph.png         (graph plot)

Requires: pandas, networkx, matplotlib
"""

import os
import math

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
from pathlib import Path

if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()

NODES_PATH = SCRIPT_DIR / "nodes.csv"
EDGES_PATH = SCRIPT_DIR / "edges_with_time.csv"
OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_distance.csv"
OUT_PNG_PATH = SCRIPT_DIR / "tehran_graph.png"

print("Project directory:", SCRIPT_DIR)
print("nodes.csv exists:", NODES_PATH.exists())
print("edges.csv exists:", EDGES_PATH.exists())


START_NODE = "n1"    # Afsarieh (example start; must exist in nodes.csv)
END_NODE = "n56"     # example end; adjust to any valid id

SHOW_EDGE_LABELS = True   # True -> draw all edge labels (cluttered!)

NODE_REQUIRED_COLS = ["id", "name", "lat", "lon"]
EDGE_REQUIRED_COLS = ["id", "from_id", "to_id", "road_name", "bidirectional", "distance_km"]


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


# ----------------------------------------------------------------------------
# Loading + validation
# ----------------------------------------------------------------------------
def load_nodes(path):
    if not os.path.exists(path):
        fail(f"nodes file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in NODE_REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(f"nodes.csv missing required columns: {missing}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        fail("node ids must be non-empty and unique")
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df.loc[df[col].isna(), "id"].tolist()
            fail(f"non-numeric/missing {col} for node ids: {bad}")
    if not df["lat"].between(-90, 90).all() or not df["lon"].between(-180, 180).all():
        fail("coordinates out of range (lat in [-90,90], lon in [-180,180])")
    df["name"] = df["name"].astype(str)
    return df


def load_edges(path):
    if not os.path.exists(path):
        fail(f"edges file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in EDGE_REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(f"edges.csv missing required columns: {missing}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        fail("edge ids must be non-empty and unique")
    df["bidirectional"] = (df["bidirectional"].astype(str).str.strip().str.lower()
                           .map({"true": True, "1": True, "yes": True,
                                 "false": False, "0": False, "no": False}))
    if df["bidirectional"].isna().any():
        fail("bidirectional column must contain True/False (or 1/0)")
    return df


# ----------------------------------------------------------------------------
# Haversine distance (vectorized)
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = math.pi / 180.0
    phi1, phi2 = lat1 * r, lat2 * r
    dphi = (lat2 - lat1) * r
    dlmb = (lon2 - lon1) * r
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fill_distances(edges, nodes):
    """Fill missing distance_km from coordinates; preserve existing values."""
    coord = nodes.set_index("id")[["lat", "lon"]]
    for col in ("from_id", "to_id"):
        unknown = set(edges[col]) - set(coord.index)
        if unknown:
            fail(f"edges reference unknown node ids in '{col}': {sorted(unknown)}")
    dist = haversine_km(edges["from_id"].map(coord["lat"]),
                        edges["from_id"].map(coord["lon"]),
                        edges["to_id"].map(coord["lat"]),
                        edges["to_id"].map(coord["lon"])).round(3)
    existing = pd.to_numeric(edges["distance_km"], errors="coerce")
    edges = edges.copy()
    edges["distance_km"] = existing.where(existing.notna(), dist)
    if (edges["distance_km"] <= 0).any() or edges["distance_km"].isna().any():
        fail("some distances are missing or non-positive")
    return edges


# ----------------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------------
def build_graph(edges, nodes):
    G = nx.DiGraph()
    for _, n in nodes.iterrows():
        G.add_node(n["id"], name=n["name"], lat=float(n["lat"]), lon=float(n["lon"]))

    for _, r in edges.iterrows():
        dist_km = float(r["distance_km"])
        attrs = {
            "edge_id": r["id"],
            "road_name": str(r["road_name"]),
            "distance_km": dist_km
        }

        if "base_traffic" in r:
            attrs["base_traffic"] = float(r["base_traffic"])
        elif "traffic" in r:
            attrs["base_traffic"] = float(r["traffic"])
        else:
            attrs["base_traffic"] = 0.0

        if "travel_time_min" in r and pd.notna(r["travel_time_min"]):
            attrs["base_time_min"] = float(r["travel_time_min"])
        else:
            attrs["base_time_min"] = (dist_km / 70.0) * 60.0

        attrs["current_time_min"] = attrs["base_time_min"]

        G.add_edge(r["from_id"], r["to_id"], **attrs)

        if bool(r["bidirectional"]):
            G.add_edge(r["to_id"], r["from_id"], **attrs)

    return G


# ----------------------------------------------------------------------------
# Plot (geographic layout: lon = x, lat = y)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Ambulance configuration
# ----------------------------------------------------------------------------

STATION_NODES = {f"n{i}" for i in range(57, 86)}

TWO_AMBULANCE_STATIONS = {
    "n57",
    "n65",
    "n68",
    "n69",
    "n74",
    "n76",
    "n77",
    "n80",
    "n81",
    "n85",
}


def create_ambulances(G):
    """
    Create the initial ambulance fleet.

    Ambulances are simulation agents, not graph nodes.
    Initially, every ambulance is located at its home station.
    """
    ambulances = []

    station_ids = sorted(
        STATION_NODES.intersection(G.nodes),
        key=lambda node_id: int(node_id[1:]),
    )

    for station_id in station_ids:
        ambulance_count = (
            2
            if station_id in TWO_AMBULANCE_STATIONS
            else 1
        )

        for number in range(1, ambulance_count + 1):
            ambulances.append({
                "id": f"AMB-{station_id[1:]}-{number}",
                "home_station": station_id,
                "current_node": station_id,
                "status": "free",
                "destination": None,
                "route": [],
                "route_index": 0,
            })

    return ambulances


# ----------------------------------------------------------------------------
# Plot (geographic layout: lon = x, lat = y)
# ----------------------------------------------------------------------------

def draw_graph(G, path, ambulances=None):
    if ambulances is None:
        ambulances = []

    pos = {
        node_id: (
            G.nodes[node_id]["lon"],
            G.nodes[node_id]["lat"],
        )
        for node_id in G.nodes
    }

    fig, ax = plt.subplots(figsize=(24, 16))

    # ------------------------------------------------------------------------
    # Separate normal nodes and ambulance-station nodes
    # ------------------------------------------------------------------------
    station_nodes = sorted(
        STATION_NODES.intersection(G.nodes),
        key=lambda node_id: int(node_id[1:]),
    )

    normal_nodes = [
        node_id
        for node_id in G.nodes
        if node_id not in STATION_NODES
    ]

    # ------------------------------------------------------------------------
    # Prepare visual edges
    # ------------------------------------------------------------------------
    visual_edges = {}

    for u, v, data in G.edges(data=True):
        edge_id = data["edge_id"]

        if edge_id not in visual_edges:
            visual_edges[edge_id] = {
                "from": u,
                "to": v,
                "road_name": data["road_name"],
                "bidirectional": False,
            }
        else:
            visual_edges[edge_id]["bidirectional"] = True

    one_way_edges = []
    two_way_edges = []
    edge_labels = {}

    for edge in visual_edges.values():
        u = edge["from"]
        v = edge["to"]

        edge_labels[(u, v)] = edge["road_name"]

        if edge["bidirectional"]:
            two_way_edges.append((u, v))
        else:
            one_way_edges.append((u, v))

    # ------------------------------------------------------------------------
    # Draw one-way edges
    # ------------------------------------------------------------------------
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=one_way_edges,
        ax=ax,
        edge_color="#8b98a5",
        width=1.1,
        alpha=0.8,
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0",
        node_size=100,
    )

    # ------------------------------------------------------------------------
    # Draw two-way edges
    # ------------------------------------------------------------------------
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=two_way_edges,
        ax=ax,
        edge_color="#8b98a5",
        width=1.1,
        alpha=0.8,
        arrows=True,
        arrowsize=12,
        arrowstyle="<|-|>",
        connectionstyle="arc3,rad=0",
        node_size=100,
    )

    # ------------------------------------------------------------------------
    # Draw normal nodes in pink
    # ------------------------------------------------------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        ax=ax,
        node_size=100,
        node_color="#ff69b4",
        edgecolors="#ad1457",
        linewidths=0.7,
        label="Neighborhood",
    )

    # ------------------------------------------------------------------------
    # Draw ambulance stations in yellow
    # ------------------------------------------------------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=station_nodes,
        ax=ax,
        node_size=190,
        node_color="#ffd54f",
        edgecolors="#f57f17",
        linewidths=1.3,
        label="115 Station",
    )

    # ------------------------------------------------------------------------
    # Draw node labels
    # ------------------------------------------------------------------------
    texts = []

    for node_id, (x, y) in pos.items():
        if node_id in STATION_NODES:
            # All station names are 115, so show the node ID as well.
            node_label = f"115\n{node_id}"
            font_size = 6
        else:
            node_label = G.nodes[node_id]["name"]
            font_size = 7

        text = ax.text(
            x,
            y,
            node_label,
            fontsize=font_size,
            ha="center",
            va="center",
            color="#202020",
            zorder=5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.70,
                "pad": 0.5,
            },
        )

        texts.append(text)

    adjust_text(
        texts,
        ax=ax,
        expand=(1.15, 1.25),
        force_text=(0.3, 0.5),
        arrowprops={
            "arrowstyle": "-",
            "color": "#999999",
            "lw": 0.4,
        },
    )

    # ------------------------------------------------------------------------
    # Draw edge labels
    # ------------------------------------------------------------------------
    if SHOW_EDGE_LABELS:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=5,
            font_color="#31465a",
            rotate=False,
            label_pos=0.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 0.4,
            },
        )

    # ------------------------------------------------------------------------
    # Group ambulances by their current node
    # ------------------------------------------------------------------------
    ambulances_by_node = {}

    for ambulance in ambulances:
        current_node = ambulance["current_node"]

        if current_node not in ambulances_by_node:
            ambulances_by_node[current_node] = []

        ambulances_by_node[current_node].append(ambulance)

    # Position offsets are measured in screen points, not longitude/latitude.
    # Therefore, the ambulance symbols stay visible at different zoom levels.
    offsets_for_one = [
        (9, -13),
    ]

    offsets_for_two = [
        (-9, -13),
        (9, -13),
    ]

    # ------------------------------------------------------------------------
    # Draw ambulances in green around their current node
    # ------------------------------------------------------------------------
    for current_node, node_ambulances in ambulances_by_node.items():
        if current_node not in pos:
            continue

        x, y = pos[current_node]

        node_ambulances = sorted(
            node_ambulances,
            key=lambda item: item["id"],
        )

        if len(node_ambulances) == 2:
            offsets = offsets_for_two
        elif len(node_ambulances) == 1:
            offsets = offsets_for_one
        else:
            # General fallback for future simulation states
            offsets = [
                ((index - (len(node_ambulances) - 1) / 2) * 11, -14)
                for index in range(len(node_ambulances))
            ]

        for index, (ambulance, offset) in enumerate(
            zip(node_ambulances, offsets),
            start=1,
        ):
            dx, dy = offset

            ax.annotate(
                str(index),
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color="white",
                zorder=8,
                bbox={
                    "boxstyle": "circle,pad=0.25",
                    "facecolor": "#2ecc71",
                    "edgecolor": "#0b6e35",
                    "linewidth": 0.8,
                    "alpha": 1.0,
                },
            )

    # ------------------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------------------
    ax.set_title(
        "Tehran Road Network and Ambulance Stations",
        fontsize=18,
        pad=16,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("auto")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.margins(0.03)

    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# Shortest route
# ----------------------------------------------------------------------------
def shortest_route(G, start, end):
    for nid in (start, end):
        if nid not in G:
            fail(f"node id {nid!r} not in graph. Valid ids: {sorted(G.nodes)}")
    cost, path = nx.single_source_dijkstra(G, start, end, weight="distance_km")
    print(f"\nShortest route: {G.nodes[start]['name']} -> {G.nodes[end]['name']}")
    print("-" * 62)
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]["distance_km"]
        total += d
        print(f"  {G.nodes[u]['name']:<16} -> {G.nodes[v]['name']:<16} "
              f"via {G[u][v]['road_name']:<18} {d:.3f} km")
    print("-" * 62)
    print(f"Total: {cost:.3f} km (segment sum {total:.3f} km, {len(path)} nodes)")
    return path, cost


def main():
    nodes = load_nodes(NODES_PATH)
    edges = load_edges(EDGES_PATH)
    edges = fill_distances(edges, nodes)
    edges.to_csv(OUT_EDGES_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_EDGES_PATH} ({len(edges)} rows)")

    G = build_graph(edges, nodes)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges")

    ambulances = create_ambulances(G)

    print("Number of stations:",
        len(STATION_NODES.intersection(G.nodes)))

    print("Number of ambulances:",
        len(ambulances))

    draw_graph(
        G,
        path="tehran_graph_with_ambulances.png",
        ambulances=ambulances,
    )


    shortest_route(G, START_NODE, END_NODE)


if __name__ == "__main__":
    main()
