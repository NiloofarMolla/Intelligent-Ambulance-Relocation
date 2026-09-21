# -*- coding: utf-8 -*-
"""
Scenario 1: Baseline with Static Traffic + Centralized Relocation.
Includes Advanced Reporting, Metrics, and Uncovered Node Visualization.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from tehran_graph import (
    SCRIPT_DIR,
    load_nodes,
    load_edges,
    build_graph,
    create_ambulances,
    STATION_NODES,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_time.csv"
INCIDENT_LOG = SCRIPT_DIR / "incident_log.csv"
COVERAGE_LOG = SCRIPT_DIR / "coverage_log.csv"
SUMMARY_LOG = SCRIPT_DIR / "response_summary.csv"
FINAL_PNG = SCRIPT_DIR / "final_state.png"
SIM_MP4 = SCRIPT_DIR / "simulation.mp4"

SAVE_MP4 = False
REALTIME = True
INTERVAL_MS = 100

SIM_START_MIN = 6 * 60
SIM_MINUTES = 24 * 60
COVERAGE_LIMIT = 8.0  # هماهنگ شده با سناریوهای دیگر

SERVICE_TIME_MIN = (25, 40)

HOTSPOT_NODES = {"n12", "n13", "n19", "n24", "n25", "n26", "n31", "n43", "n45"}
HOTSPOT_MULT = 1.8

# افزایش نرخ تقاضا به حدود 400 حادثه در روز
RATES_PER_HOUR = {
    (0, 6): 6,
    (6, 9): 25,
    (9, 12): 15,
    (12, 16): 15,
    (16, 20): 25,
    (20, 24): 22,
}

COL_NORMAL = "#ff69b4"
COL_STATION = "#ffd54f"
COL_MILD = "#ff9999"
COL_SEVERE = "#cc0000"
COL_SERVICING = "#a9a9a9"
COL_UNCOVERED = "#d2b48c"
COL_AMB = "#2ecc71"


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def fmt_clock(minute_of_day):
    if minute_of_day is None:
        return ""
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


# ----------------------------------------------------------------------------
# Precomputing Shortest Paths (Static Traffic)
# ----------------------------------------------------------------------------
def precompute_paths(G):
    print("Precomputing shortest paths to optimize simulation...")
    times = dict(nx.all_pairs_dijkstra_path_length(G, weight="travel_time_min"))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight="travel_time_min"))
    return times, paths


# ----------------------------------------------------------------------------
# Incident generation
# ----------------------------------------------------------------------------
def plan_incidents(G, rng):
    schedule = {t: [] for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES)}
    eligible_nodes = [n for n in G.nodes if n not in STATION_NODES]

    weights = {n: (HOTSPOT_MULT if n in HOTSPOT_NODES else 1.0) for n in eligible_nodes}
    total_weight = sum(weights.values())

    next_id = 1
    total_planned = 0

    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        hour = (t // 60) % 24
        current_rate_hr = 10
        for (h_start, h_end), rate in RATES_PER_HOUR.items():
            if h_start <= hour < h_end:
                current_rate_hr = rate
                break

        city_rate_min = current_rate_hr / 60.0

        for node in eligible_nodes:
            node_lambda = city_rate_min * (weights[node] / total_weight)
            count = rng.poisson(node_lambda)
            for _ in range(count):
                severity = "severe" if rng.random() < 0.35 else "mild"
                deadline = 8.0 if severity == "severe" else 12.0

                inc = {
                    "id": f"INC-{next_id:04d}",
                    "node": node,
                    "created_min": t,
                    "created_clock": fmt_clock(t),
                    "severity": severity,
                    "deadline_min": deadline,
                    "status": "waiting",
                    "assigned": None,
                    "arrival_min": None,
                    "arrival_clock": None,
                    "service_end_min": None,
                    "service_end_clock": None,
                    "white_at": None
                }
                schedule[t].append(inc)
                next_id += 1
                total_planned += 1

    print(f"Total incidents planned for the day (High Demand): {total_planned}")
    return schedule


# ----------------------------------------------------------------------------
# Ambulance movement helpers
# ----------------------------------------------------------------------------
def assign_mission(amb, incident, t, kind, times, paths):
    target = incident["node"]
    if target not in paths[amb["current_node"]]:
        return False

    amb["route"] = paths[amb["current_node"]][target]
    amb["route_index"] = 0
    amb["edge_elapsed"] = 0.0
    amb["target_node"] = target
    amb["incident_id"] = incident["id"] if kind == "dispatch" else None
    amb["status"] = "busy" if kind == "dispatch" else "relocating"
    amb["_arrived_handled"] = False

    if kind == "dispatch":
        incident["status"] = "assigned"
        incident["assigned"] = amb["id"]
    return True


def step_movement(G, ambulances):
    for amb in ambulances:
        if amb["status"] not in ("busy", "relocating"):
            continue

        remaining = 1.0
        while remaining > 1e-9 and amb["route_index"] < len(amb["route"]) - 1:
            u = amb["route"][amb["route_index"]]
            v = amb["route"][amb["route_index"] + 1]
            edge_t = G[u][v]["travel_time_min"]
            left = edge_t - amb["edge_elapsed"]
            if remaining < left:
                amb["edge_elapsed"] += remaining
                remaining = 0.0
            else:
                remaining -= left
                amb["route_index"] += 1
                amb["edge_elapsed"] = 0.0
                amb["current_node"] = v

        if amb["route_index"] >= len(amb["route"]) - 1:
            amb["current_node"] = amb["route"][-1]
            amb["route"] = [amb["current_node"]]
            amb["route_index"] = 0
            amb["edge_elapsed"] = 0.0

            if not amb.get("_arrived_handled"):
                amb["_arrived_handled"] = True
                if amb["status"] == "busy":
                    yield ("arrived", amb)
                else:
                    amb["status"] = "free"
                    amb["target_node"] = None
                    yield ("relocated", amb)


# ----------------------------------------------------------------------------
# Coverage (Logical vs Actual)
# ----------------------------------------------------------------------------
def get_coverage_states(G, ambulances, times):
    free_positions = [a["current_node"] for a in ambulances if a["status"] == "free"]

    effective_positions = list(free_positions)
    for a in ambulances:
        if a["status"] == "relocating" and a.get("target_node"):
            effective_positions.append(a["target_node"])

    actually_uncovered = set()
    logically_uncovered = set()

    for node in G.nodes:
        if node in STATION_NODES:
            continue

        best_actual = min((times[p].get(node, np.inf) for p in free_positions), default=np.inf)
        if best_actual > COVERAGE_LIMIT:
            actually_uncovered.add(node)

        best_logical = min((times[p].get(node, np.inf) for p in effective_positions), default=np.inf)
        if best_logical > COVERAGE_LIMIT:
            logically_uncovered.add(node)

    return actually_uncovered, logically_uncovered


def node_color(node, incidents, t, uncovered_set):
    if node in STATION_NODES:
        return COL_STATION

    active = [i for i in incidents if
              i["node"] == node and i["status"] in ("waiting", "assigned", "arrived_local", "serving")]
    if active:
        inc = active[-1]
        if inc["status"] in ("waiting", "assigned", "arrived_local"):
            return COL_SEVERE if inc["severity"] == "severe" else COL_MILD
        return COL_SERVICING

    if node in uncovered_set:
        return COL_UNCOVERED

    return COL_NORMAL


# ----------------------------------------------------------------------------
# Main simulation
# ----------------------------------------------------------------------------
def run_simulation(G, pos, ambulances, rng, times, paths):
    incidents = []
    incident_schedule = plan_incidents(G, rng)
    coverage_rows = []
    response_rows = []
    events = []

    uncovered_since = {}
    uncovered_intervals = []
    history_frames = []

    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        clock = fmt_clock(t % (24 * 60))
        hour = (t // 60) % 24

        # Night Sweep
        if t % (24 * 60) == 0:
            for amb in ambulances:
                if amb["status"] == "free" and amb["current_node"] != amb["home_station"]:
                    fake = {"node": amb["home_station"]}
                    assign_mission(amb, fake, t, "relocate", times, paths)
                    events.append((t, f"{clock}  NIGHT SHIFT: {amb['id']} -> home"))

        for inc in incident_schedule.get(t, []):
            incidents.append(inc)
            events.append((t, f"{clock}  {inc['id']} {inc['severity']} @ {inc['node']}"))

        waiting = [i for i in incidents if i["status"] == "waiting"]
        waiting.sort(key=lambda i: (i["severity"] != "severe", i["created_min"]))

        for inc in waiting:
            free = [a for a in ambulances if a["status"] == "free"]
            if not free:
                break

            best, best_time = min(
                ((a, times[a["current_node"]].get(inc["node"], np.inf)) for a in free),
                key=lambda x: x[1], default=(None, np.inf)
            )

            if best is None or best_time == np.inf:
                continue

            if best["current_node"] == inc["node"]:
                best["status"] = "busy"
                best["route"] = [inc["node"]]
                best["route_index"] = 0
                best["edge_elapsed"] = 0.0
                best["target_node"] = inc["node"]
                best["incident_id"] = inc["id"]
                best["_arrived_handled"] = True

                inc["status"] = "arrived_local"
                inc["assigned"] = best["id"]
                inc["arrival_min"] = t
                inc["arrival_clock"] = fmt_clock(t)
                inc["white_at"] = t + 2

                response_time = t - inc["created_min"]
                on_time = response_time <= inc["deadline_min"]

                service_duration = int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
                best["service_end_min"] = t + 2 + service_duration

                response_rows.append({
                    "incident_id": inc["id"], "node": inc["node"], "severity": inc["severity"],
                    "created_clock": inc["created_clock"], "arrival_clock": inc["arrival_clock"],
                    "response_time_min": response_time, "deadline_min": inc["deadline_min"],
                    "service_duration_min": service_duration, "on_time": on_time, "ambulance": best["id"],
                })
                tag = "" if on_time else "  ** FAIL"
                events.append((t, f"{clock}  {inc['id']}: local arrival in {response_time:.0f}m{tag}"))

            else:
                if assign_mission(best, inc, t, "dispatch", times, paths):
                    events.append((t, f"{clock}  dispatch {best['id']} -> {inc['id']}"))

        for inc in incidents:
            if inc["status"] == "arrived_local" and t >= inc["white_at"]:
                inc["status"] = "serving"

        for ev, amb in step_movement(G, ambulances):
            if ev == "arrived":
                inc = next(i for i in incidents if i["id"] == amb["incident_id"])
                arrival_min = t + 1
                inc["arrival_min"] = arrival_min
                inc["arrival_clock"] = fmt_clock(arrival_min)
                inc["status"] = "serving"

                rt = arrival_min - inc["created_min"]
                on_time = rt <= inc["deadline_min"]

                service_duration = int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
                amb["service_end_min"] = arrival_min + service_duration

                response_rows.append({
                    "incident_id": inc["id"], "node": inc["node"], "severity": inc["severity"],
                    "created_clock": inc["created_clock"], "arrival_clock": inc["arrival_clock"],
                    "response_time_min": rt, "deadline_min": inc["deadline_min"],
                    "service_duration_min": service_duration, "on_time": on_time, "ambulance": amb["id"],
                })
                tag = "" if on_time else "  ** FAIL"
                events.append((t, f"{clock}  {inc['id']}: arrived in {rt:.0f}m{tag}"))

        for amb in ambulances:
            if amb["status"] == "busy" and amb.get("service_end_min") is not None and t >= amb["service_end_min"]:
                inc = next(i for i in incidents if i["id"] == amb["incident_id"])
                inc["status"] = "done"
                inc["service_end_min"] = t
                inc["service_end_clock"] = fmt_clock(t)
                amb["service_end_min"] = None
                amb["incident_id"] = None

                if 0 <= hour < 6:
                    if amb["current_node"] != amb["home_station"]:
                        fake = {"node": amb["home_station"]}
                        assign_mission(amb, fake, t, "relocate", times, paths)
                    else:
                        amb["status"] = "free"
                else:
                    amb["status"] = "free"

        # ----------------------------------------------------
        # Centralized Advanced Relocation (Like Scenarios 2/3)
        # ----------------------------------------------------
        if 6 <= hour < 24:
            _, logically_bad = get_coverage_states(G, ambulances, times)

            if logically_bad:
                for target in list(logically_bad):
                    free_ambs = [a for a in ambulances if a["status"] == "free"]
                    if not free_ambs:
                        break

                    best_amb = None
                    best_time = np.inf

                    for amb in free_ambs:
                        d = times[amb["current_node"]].get(target, np.inf)
                        if d >= best_time:
                            continue

                        # Safety check
                        others_positions = []
                        for a in ambulances:
                            if a is not amb:
                                if a["status"] == "free":
                                    others_positions.append(a["current_node"])
                                elif a["status"] == "relocating" and a.get("target_node"):
                                    others_positions.append(a["target_node"])

                        safe = True
                        for node in G.nodes:
                            if node in STATION_NODES or node == target:
                                continue

                            covered_by_amb = times[amb["current_node"]].get(node, np.inf) <= COVERAGE_LIMIT

                            if covered_by_amb:
                                covered_by_others = False
                                if others_positions:
                                    best_other = min((times[p].get(node, np.inf) for p in others_positions),
                                                     default=np.inf)
                                    if best_other <= COVERAGE_LIMIT:
                                        covered_by_others = True

                                if not covered_by_others:
                                    covered_from_target = times[target].get(node, np.inf) <= COVERAGE_LIMIT
                                    if not covered_from_target:
                                        safe = False
                                        break

                        if safe:
                            best_amb = amb
                            best_time = d

                    if best_amb is not None:
                        fake = {"node": target}
                        if best_amb["current_node"] != target:
                            if assign_mission(best_amb, fake, t, "relocate", times, paths):
                                events.append((t, f"{clock}  relocate {best_amb['id']} -> {target}"))
                                logically_bad.remove(target)

        # ----------------------------------------------------
        # Coverage Log (Actual)
        # ----------------------------------------------------
        actually_bad, _ = get_coverage_states(G, ambulances, times)

        for node in list(uncovered_since):
            if node not in actually_bad:
                start_t = uncovered_since.pop(node)
                duration = t - start_t
                if duration > 0:
                    uncovered_intervals.append(
                        {"node": node, "start_min": start_t, "end_min": t, "duration_min": duration})

        for node in actually_bad:
            if node not in uncovered_since:
                uncovered_since[node] = t

        coverage_rows.append({
            "clock": clock,
            "uncovered_count": len(actually_bad), "uncovered_nodes": ";".join(sorted(actually_bad)),
            "free": sum(a["status"] == "free" for a in ambulances),
            "busy": sum(a["status"] == "busy" for a in ambulances),
            "relocating": sum(a["status"] == "relocating" for a in ambulances),
        })

        frame_nodes = [node_color(n, incidents, t, actually_bad) for n in G.nodes]
        frame_ambs = []
        for a in ambulances:
            if a["status"] in ("busy", "relocating") and len(a["route"]) > 1:
                u = a["route"][a["route_index"]]
                v = a["route"][a["route_index"] + 1]
                edge_t = G[u][v].get("travel_time_min", 1.0)
                alpha = min(1.0, a["edge_elapsed"] / edge_t) if edge_t > 0 else 1.0
                x = (1.0 - alpha) * pos[u][0] + alpha * pos[v][0]
                y = (1.0 - alpha) * pos[u][1] + alpha * pos[v][1]
            else:
                cx, cy = pos[a["current_node"]]
                idx = int(a["id"].split("-")[-1])
                dx = (idx % 3 - 1) * 0.003
                dy = ((idx // 3) % 3 - 1) * 0.003
                x, y = cx + dx, cy + dy
            frame_ambs.append({"id": a["id"], "x": x, "y": y, "status": a["status"]})

        history_frames.append({"nodes": frame_nodes, "ambs": frame_ambs})

    end_t = SIM_START_MIN + SIM_MINUTES
    for node, start_t in uncovered_since.items():
        duration = end_t - start_t
        if duration > 0:
            uncovered_intervals.append({"node": node, "start_min": start_t, "end_min": end_t, "duration_min": duration})

    return incidents, coverage_rows, response_rows, events, history_frames, uncovered_intervals


# ----------------------------------------------------------------------------
# Generate KPI Reports & Charts
# ----------------------------------------------------------------------------
def generate_reports(incidents, coverage_rows, response_rows, uncovered_intervals):
    print("\n--- Generating Defense Reports and Charts ---")

    df_resp = pd.DataFrame(response_rows)
    if not df_resp.empty:
        total_missions = len(df_resp)
        success_missions = df_resp['on_time'].sum()
        fail_missions = total_missions - success_missions
        on_time_rate = (success_missions / total_missions) * 100
        avg_response = df_resp['response_time_min'].mean()
        avg_service = df_resp['service_duration_min'].mean()

        kpi_data = {
            "Metric": [
                "Total Missions of the Day",
                "Successful Missions (On Time)",
                "Failed Missions (Late)",
                "On-Time Response Rate (%)",
                "Average Response Time (min)",
                "Average Service Time at Scene (min)"
            ],
            "Value": [
                total_missions,
                success_missions,
                fail_missions,
                f"{on_time_rate:.2f}%",
                f"{avg_response:.2f}",
                f"{avg_service:.2f}"
            ]
        }
        pd.DataFrame(kpi_data).to_csv(SCRIPT_DIR / "KPI_summary.csv", index=False)

    df_unc = pd.DataFrame(uncovered_intervals)
    if not df_unc.empty:
        node_stats = df_unc.groupby('node').agg(
            Frequency=('node', 'count'),
            Average_Duration_Min=('duration_min', 'mean'),
            Total_Duration_Min=('duration_min', 'sum')
        ).reset_index().round(2)
        node_stats.to_csv(SCRIPT_DIR / "Node_Coverage_Stats.csv", index=False)

    if not df_resp.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(df_resp['response_time_min'], bins=20, color='skyblue', edgecolor='black')
        plt.axvline(8, color='red', linestyle='dashed', linewidth=2, label='Severe Deadline (8m)')
        plt.axvline(12, color='orange', linestyle='dashed', linewidth=2, label='Mild Deadline (12m)')
        plt.title('Distribution of Ambulance Response Times', fontsize=14)
        plt.xlabel('Response Time (minutes)', fontsize=12)
        plt.ylabel('Number of Incidents', fontsize=12)
        plt.legend()
        plt.savefig(SCRIPT_DIR / "chart_response_times.png", dpi=150, bbox_inches='tight')
        plt.close()

    df_cov = pd.DataFrame(coverage_rows)
    if not df_cov.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(df_cov)), df_cov['uncovered_count'], color='brown', linewidth=2)
        plt.title('Number of Uncovered Nodes over 24 Hours', fontsize=14)
        plt.xlabel('Simulation Minute (0 = 06:00)', fontsize=12)
        plt.ylabel('Count of Uncovered Nodes', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(SCRIPT_DIR / "chart_uncovered_nodes.png", dpi=150, bbox_inches='tight')
        plt.close()

    df_inc = pd.DataFrame(incidents)
    if not df_inc.empty:
        df_inc['hour'] = df_inc['created_clock'].str.split(':').str[0].astype(int)
        hourly_counts = df_inc['hour'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        hourly_counts.plot(kind='bar', color='#ff9999', edgecolor='black')
        plt.title('Hourly Incident Frequency', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Incidents', fontsize=12)
        plt.xticks(rotation=0)
        plt.savefig(SCRIPT_DIR / "chart_hourly_incidents.png", dpi=150, bbox_inches='tight')
        plt.close()


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def make_anim(G, pos, ambulances, coverage_rows, events, history_frames):
    fig, ax = plt.subplots(figsize=(24, 16))

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#8b98a5", width=1.0, alpha=0.8, node_size=100)
    node_scatter = ax.scatter([pos[n][0] for n in G.nodes], [pos[n][1] for n in G.nodes], s=110, zorder=4)

    for n in G.nodes:
        name = G.nodes[n].get("name", str(n))
        if n in STATION_NODES:
            name = f"115\n{n}"
        ax.text(pos[n][0], pos[n][1] + 0.0018, name, fontsize=6, ha='center', va='bottom', color='#2b2b2b', zorder=5)

    amb_scatters = {}
    for a in ambulances:
        point, = ax.plot([], [], 'o', color=COL_AMB, markersize=8, markeredgecolor='#0b6e35', zorder=8)
        amb_scatters[a["id"]] = point

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_NORMAL, markersize=10,
               label='Covered Neighborhood'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_UNCOVERED, markersize=10,
               label='Uncovered Neighborhood (>8m)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_STATION, markersize=10, label='115 Station (Base)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_MILD, markersize=10, label='Mild Incident'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_SEVERE, markersize=10, label='Severe Incident'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_SERVICING, markersize=10, label='Servicing Scene'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_AMB, markersize=10, markeredgecolor='#0b6e35',
               label='Ambulance'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, title="Simulation Guide")

    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, fontsize=12, va="top", family="monospace",
                   bbox={"facecolor": "white", "alpha": 0.85})
    logbox = ax.text(0.01, 0.80, "", transform=ax.transAxes, fontsize=8, va="top", family="monospace",
                     bbox={"facecolor": "white", "alpha": 0.7})

    ax.set_aspect("auto")
    ax.margins(0.03)
    ax.set_title("Ambulance Dispatch & Relocation Simulation (Agent Baseline)", fontsize=18)

    def update(frame):
        t = SIM_START_MIN + frame
        clock = fmt_clock(t % (24 * 60))
        row = coverage_rows[frame]
        h = history_frames[frame]

        node_scatter.set_color(h["nodes"])

        for a_data in h["ambs"]:
            amb = amb_scatters[a_data["id"]]
            amb.set_data([a_data["x"]], [a_data["y"]])
            amb.set_alpha(1.0 if a_data["status"] == "free" else 0.85)

        info.set_text(
            f"Time: {clock}   |   Free: {row['free']}  Busy: {row['busy']}  Reloc: {row['relocating']}\nUncovered nodes: {row['uncovered_count']}")
        recent = [e for _, e in events if e.startswith(clock)]
        logbox.set_text("\n".join(recent[-8:]) if recent else "")

        return [node_scatter, info, logbox] + list(amb_scatters.values())

    ani = animation.FuncAnimation(fig, update, frames=SIM_MINUTES, interval=INTERVAL_MS if REALTIME else 1, blit=False,
                                  repeat=False)
    return fig, ani


def main():
    nodes = load_nodes(SCRIPT_DIR / "nodes.csv")
    edges = load_edges(OUT_EDGES_PATH)
    if "travel_time_min" not in edges.columns:
        fail("edges file must contain travel_time_min")

    G = build_graph(edges, nodes)
    for u, v in G.edges:
        G[u][v].setdefault("travel_time_min", G[u][v]["distance_km"] / 70.0 * 60.0)

    ambulances = create_ambulances(G)
    print(f"Graph: {G.number_of_nodes()} nodes | Ambulances: {len(ambulances)}")

    pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}
    times, paths = precompute_paths(G)
    rng = np.random.default_rng(42)

    incidents, coverage_rows, response_rows, events, history_frames, uncovered_intervals = run_simulation(G, pos,
                                                                                                          ambulances,
                                                                                                          rng, times,
                                                                                                          paths)

    pd.DataFrame(incidents)[
        ["id", "node", "created_clock", "severity", "status", "arrival_clock", "service_end_clock"]
    ].to_csv(INCIDENT_LOG, index=False)

    pd.DataFrame(coverage_rows).to_csv(COVERAGE_LOG, index=False)

    resp = pd.DataFrame(response_rows)
    resp.to_csv(SUMMARY_LOG, index=False)

    generate_reports(incidents, coverage_rows, response_rows, uncovered_intervals)

    fig, ani = make_anim(G, pos, ambulances, coverage_rows, events, history_frames)

    if SAVE_MP4:
        writer = animation.FFMpegWriter(fps=20, bitrate=2000)
        ani.save(SIM_MP4, writer=writer)
        print(f"\nSUCCESS! Video saved successfully at: {SIM_MP4}")
    else:
        plt.show()


if __name__ == "__main__":
    main()