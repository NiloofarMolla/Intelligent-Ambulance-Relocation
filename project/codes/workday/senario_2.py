# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import os

from tehran_graph import (
    SCRIPT_DIR, load_nodes, load_edges, build_graph, create_ambulances, STATION_NODES
)

# Config
OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_traffic.csv"
INCIDENT_LOG = SCRIPT_DIR / "incident_log_2.csv"
COVERAGE_LOG = SCRIPT_DIR / "coverage_log_2.csv"
SUMMARY_LOG = SCRIPT_DIR / "response_summary_2.csv"
OUT_OF_SERVICE_LOG = SCRIPT_DIR / "out_of_service_log_2.csv"
SIM_MP4 = SCRIPT_DIR / "simulation_2.mp4"

SAVE_MP4 = False
REALTIME = False
INTERVAL_MS = 100
NUM_SEEDS = 30  

SIM_START_MIN = 6 * 60
SIM_MINUTES = 24 * 60
COVERAGE_LIMIT = 8.0

SERVICE_TIME_MIN = (25, 40)
P_BREAKDOWN_PER_MIN = 0.0005
OUT_OF_SERVICE_TIME_MIN = (30, 40)
HOTSPOT_NODES = {"n12", "n13", "n19", "n24", "n25", "n26", "n31", "n43", "n45"}
HOTSPOT_MULT = 1.8

RATES_PER_HOUR = {
    (0, 6): 6, (6, 9): 25, (9, 12): 15,
    (12, 16): 15, (16, 20): 25, (20, 24): 22,
}

COL_NORMAL, COL_STATION = "#ff69b4", "#ffd54f"
COL_MILD, COL_SEVERE = "#ff9999", "#cc0000"
COL_SERVICING, COL_UNCOVERED = "#a9a9a9", "#d2b48c"
COL_AMB, COL_OUT = "#2ecc71", "#000000"


def fmt_clock(minute_of_day):
    if minute_of_day is None: return ""
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def get_time_multiplier(hour):
    if 0 <= hour < 6:
        return 0.1
    elif 6 <= hour < 9:
        return 0.9
    elif 9 <= hour < 12:
        return 0.5
    elif 12 <= hour < 16:
        return 0.6
    elif 16 <= hour < 20:
        return 0.9
    elif 20 <= hour < 24:
        return 0.4
    return 0.5


def update_traffic_and_paths(G, hour):
    mult = get_time_multiplier(hour)
    for u, v, data in G.edges(data=True):
        C = float(data.get('base_traffic', 0.0)) * mult
        if C < 0.4:
            data['color'], data['current_time_min'] = "#3498db", data['base_time_min'] * (1.0 + C)
        elif C < 0.6:
            data['color'], data['current_time_min'] = "#e67e22", data['base_time_min'] * (2.0 + C)
        else:
            data['color'], data['current_time_min'] = "#9b59b6", data['base_time_min'] * (3.0 + C)
    return dict(nx.all_pairs_dijkstra_path_length(G, weight="current_time_min")), dict(
        nx.all_pairs_dijkstra_path(G, weight="current_time_min"))


def plan_incidents(G, rng):
    schedule = {t: [] for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES)}
    eligible_nodes = [n for n in G.nodes if n not in STATION_NODES]
    weights = {n: (HOTSPOT_MULT if n in HOTSPOT_NODES else 1.0) for n in eligible_nodes}
    tot_weight = sum(weights.values())
    next_id = 1
    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        hour = (t // 60) % 24
        rate = next(r for (h_s, h_e), r in RATES_PER_HOUR.items() if h_s <= hour < h_e)
        city_rate_min = rate / 60.0
        for node in eligible_nodes:
            count = rng.poisson(city_rate_min * (weights[node] / tot_weight))
            for _ in range(count):
                sev = "severe" if rng.random() < 0.35 else "mild"
                schedule[t].append({
                    "id": f"INC-{next_id:04d}", "node": node, "created_min": t, "created_clock": fmt_clock(t),
                    "severity": sev, "deadline_min": 8.0 if sev == "severe" else 12.0, "status": "waiting",
                    "assigned": None, "arrival_min": None, "arrival_clock": None, "service_end_min": None,
                    "service_end_clock": None, "white_at": None
                })
                next_id += 1
    return schedule


def assign_mission(amb, incident, t, kind, times, paths):
    target = incident["node"]
    if target not in paths[amb["current_node"]]: return False
    amb.update({"route": paths[amb["current_node"]][target], "route_index": 0, "edge_elapsed": 0.0,
                "target_node": target, "incident_id": incident["id"] if kind == "dispatch" else None,
                "status": "busy" if kind == "dispatch" else "relocating", "_arrived_handled": False})
    if kind == "dispatch": incident.update({"status": "assigned", "assigned": amb["id"]})
    return True


def step_movement(G, ambulances):
    for amb in ambulances:
        if amb["status"] not in ("busy", "relocating"): continue
        rem = 1.0
        while rem > 1e-9 and amb["route_index"] < len(amb["route"]) - 1:
            u, v = amb["route"][amb["route_index"]], amb["route"][amb["route_index"] + 1]
            left = G[u][v]["current_time_min"] - amb["edge_elapsed"]
            if rem < left:
                amb["edge_elapsed"] += rem; rem = 0.0
            else:
                rem -= left; amb["route_index"] += 1; amb["edge_elapsed"] = 0.0; amb["current_node"] = v
        if amb["route_index"] >= len(amb["route"]) - 1:
            amb.update(
                {"current_node": amb["route"][-1], "route": [amb["route"][-1]], "route_index": 0, "edge_elapsed": 0.0})
            if not amb.get("_arrived_handled"):
                amb["_arrived_handled"] = True
                if amb["status"] == "busy":
                    yield ("arrived", amb)
                else:
                    amb.update({"status": "free", "target_node": None}); yield ("relocated", amb)


def get_coverage_states(G, ambulances, times):
    free = [a["current_node"] for a in ambulances if a["status"] == "free"]
    eff = free + [a["target_node"] for a in ambulances if a["status"] == "relocating" and a.get("target_node")]
    act, log = set(), set()
    for n in G.nodes:
        if n in STATION_NODES: continue
        if min((times[p].get(n, np.inf) for p in free), default=np.inf) > COVERAGE_LIMIT: act.add(n)
        if min((times[p].get(n, np.inf) for p in eff), default=np.inf) > COVERAGE_LIMIT: log.add(n)
    return act, log


def node_color(node, incidents, t, uncovered_set):
    if node in STATION_NODES: return COL_STATION
    act = [i for i in incidents if
           i["node"] == node and i["status"] in ("waiting", "assigned", "arrived_local", "serving")]
    if act:
        inc = act[-1]
        if inc["status"] in ("waiting", "assigned", "arrived_local"): return COL_SEVERE if inc[
                                                                                               "severity"] == "severe" else COL_MILD
        return COL_SERVICING
    if node in uncovered_set: return COL_UNCOVERED
    return COL_NORMAL


def run_simulation(G, pos, ambulances, rng, record_history=False):
    incidents = []
    incident_schedule = plan_incidents(G, rng)
    cov_rows, resp_rows, out_rows, events, history_frames, uncov_intervals = [], [], [], [], [], []
    uncov_since = {}
    times, paths, last_mult = None, None, -1

    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        clock, hour = fmt_clock(t % (24 * 60)), (t // 60) % 24
        cur_mult = get_time_multiplier(hour)
        if cur_mult != last_mult:
            if record_history: events.append((t, f"--- TRAFFIC UPDATE: Hour {hour:02d} ---"))
            times, paths = update_traffic_and_paths(G, hour)
            last_mult = cur_mult

        for amb in ambulances:
            if amb["status"] == "out_of_service" and t >= amb.get("out_of_service_until", 0):
                amb.update({"status": "free", "out_of_service_until": None})

        if t % (24 * 60) == 0:
            for amb in ambulances:
                if amb["status"] == "free" and amb["current_node"] != amb["home_station"]:
                    assign_mission(amb, {"node": amb["home_station"]}, t, "relocate", times, paths)

        for amb in ambulances:
            if amb["status"] == "free" and rng.random() < P_BREAKDOWN_PER_MIN:
                dt = int(rng.integers(OUT_OF_SERVICE_TIME_MIN[0], OUT_OF_SERVICE_TIME_MIN[1] + 1))
                amb.update({"status": "out_of_service", "out_of_service_until": t + dt})
                out_rows.append({"ambulance_id": amb["id"], "location_node": amb["current_node"], "start_min": t,
                                 "start_clock": clock, "duration_min": dt, "end_min": t + dt,
                                 "end_clock": fmt_clock(t + dt), "reason": "Breakdown/Fatigue"})

        for inc in incident_schedule.get(t, []):
            incidents.append(inc)

        waiting = sorted([i for i in incidents if i["status"] == "waiting"],
                         key=lambda x: (x["severity"] != "severe", x["created_min"]))
        for inc in waiting:
            free = [a for a in ambulances if a["status"] == "free"]
            if not free: break
            best, b_time = min(((a, times[a["current_node"]].get(inc["node"], np.inf)) for a in free),
                               key=lambda x: x[1], default=(None, np.inf))
            if best is None or b_time == np.inf: continue

            if best["current_node"] == inc["node"]:
                best.update({"status": "busy", "route": [inc["node"]], "route_index": 0, "edge_elapsed": 0.0,
                             "target_node": inc["node"], "incident_id": inc["id"], "_arrived_handled": True})
                inc.update({"status": "arrived_local", "assigned": best["id"], "arrival_min": t, "arrival_clock": clock,
                            "white_at": t + 2})
                rt = t - inc["created_min"]
                sd = int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
                best["service_end_min"] = t + 2 + sd
                resp_rows.append({"incident_id": inc["id"], "node": inc["node"], "severity": inc["severity"],
                                  "created_clock": inc["created_clock"], "arrival_clock": inc["arrival_clock"],
                                  "response_time_min": rt, "deadline_min": inc["deadline_min"],
                                  "service_duration_min": sd, "on_time": rt <= inc["deadline_min"],
                                  "ambulance": best["id"]})
            else:
                assign_mission(best, inc, t, "dispatch", times, paths)

        for inc in incidents:
            if inc["status"] == "arrived_local" and t >= inc["white_at"]: inc["status"] = "serving"

        for ev, amb in step_movement(G, ambulances):
            if ev == "arrived":
                inc = next(i for i in incidents if i["id"] == amb["incident_id"])
                inc.update({"arrival_min": t + 1, "arrival_clock": fmt_clock(t + 1), "status": "serving"})
                rt = t + 1 - inc["created_min"]
                sd = int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
                amb["service_end_min"] = t + 1 + sd
                resp_rows.append({"incident_id": inc["id"], "node": inc["node"], "severity": inc["severity"],
                                  "created_clock": inc["created_clock"], "arrival_clock": inc["arrival_clock"],
                                  "response_time_min": rt, "deadline_min": inc["deadline_min"],
                                  "service_duration_min": sd, "on_time": rt <= inc["deadline_min"],
                                  "ambulance": amb["id"]})

        for amb in ambulances:
            if amb["status"] == "busy" and amb.get("service_end_min") is not None and t >= amb["service_end_min"]:
                inc = next(i for i in incidents if i["id"] == amb["incident_id"])
                inc.update({"status": "done", "service_end_min": t, "service_end_clock": clock})
                amb.update({"service_end_min": None, "incident_id": None})
                if 0 <= hour < 6:
                    if amb["current_node"] != amb["home_station"]:
                        assign_mission(amb, {"node": amb["home_station"]}, t, "relocate", times, paths)
                    else:
                        amb["status"] = "free"
                else:
                    amb["status"] = "free"

        if 6 <= hour < 24:
            _, log_bad = get_coverage_states(G, ambulances, times)
            if log_bad:
                for target in sorted(list(log_bad)):
                    free_ambs = [a for a in ambulances if a["status"] == "free"]
                    if not free_ambs: break
                    best_amb, best_time = None, np.inf
                    for amb in free_ambs:
                        d = times[amb["current_node"]].get(target, np.inf)
                        if d >= best_time: continue
                        others = [a["current_node"] if a["status"] == "free" else a["target_node"] for a in ambulances
                                  if a is not amb and a["status"] in ("free", "relocating") and a.get("target_node")]
                        safe = True
                        for n in G.nodes:
                            if n in STATION_NODES or n == target: continue
                            if times[amb["current_node"]].get(n, np.inf) <= COVERAGE_LIMIT:
                                if not others or min((times[p].get(n, np.inf) for p in others),
                                                     default=np.inf) > COVERAGE_LIMIT:
                                    if times[target].get(n, np.inf) > COVERAGE_LIMIT:
                                        safe = False;
                                        break
                        if safe: best_amb, best_time = amb, d
                    if best_amb and best_amb["current_node"] != target:
                        if assign_mission(best_amb, {"node": target}, t, "relocate", times, paths): log_bad.remove(
                            target)

        act_bad, _ = get_coverage_states(G, ambulances, times)
        for n in list(uncov_since):
            if n not in act_bad:
                st = uncov_since.pop(n)
                if t - st > 0: uncov_intervals.append(
                    {"node": n, "start_min": st, "end_min": t, "duration_min": t - st})
        for n in act_bad:
            if n not in uncov_since: uncov_since[n] = t

        cov_rows.append({"minute": t, "clock": clock, "uncovered_count": len(act_bad),
                         "free": sum(a["status"] == "free" for a in ambulances),
                         "busy": sum(a["status"] == "busy" for a in ambulances),
                         "relocating": sum(a["status"] == "relocating" for a in ambulances),
                         "out_of_service": sum(a["status"] == "out_of_service" for a in ambulances)})

        if record_history:
            f_nodes = [node_color(n, incidents, t, act_bad) for n in G.nodes]
            f_edges = [data.get('color', '#8b98a5') for u, v, data in G.edges(data=True)]
            f_ambs = []
            for a in ambulances:
                if a["status"] in ("busy", "relocating") and len(a["route"]) > 1:
                    u, v = a["route"][a["route_index"]], a["route"][a["route_index"] + 1]
                    edge_t = G[u][v].get("current_time_min", 1.0)
                    alpha = min(1.0, a["edge_elapsed"] / edge_t) if edge_t > 0 else 1.0
                    x = (1.0 - alpha) * pos[u][0] + alpha * pos[v][0]
                    y = (1.0 - alpha) * pos[u][1] + alpha * pos[v][1]
                else:
                    cx, cy = pos[a["current_node"]]
                    idx = int(a["id"].split("-")[-1])
                    x, y = cx + (idx % 3 - 1) * 0.003, cy + ((idx // 3) % 3 - 1) * 0.003
                f_ambs.append({"id": a["id"], "x": x, "y": y, "status": a["status"]})
            history_frames.append({"nodes": f_nodes, "edges": f_edges, "ambs": f_ambs})

    end_t = SIM_START_MIN + SIM_MINUTES
    for n, st in uncov_since.items():
        if end_t - st > 0: uncov_intervals.append(
            {"node": n, "start_min": st, "end_min": end_t, "duration_min": end_t - st})
    return incidents, cov_rows, resp_rows, events, history_frames, uncov_intervals, out_rows


def generate_reports(df_resp, df_cov, df_unc):
    print("\n--- Generating Scenario 2 Reports (30-Day Avg) ---")
    if not df_resp.empty:
        tot = len(df_resp) / NUM_SEEDS
        suc = df_resp['on_time'].sum() / NUM_SEEDS
        fail = tot - suc
        rate = (suc / tot) * 100 if tot > 0 else 0
        pd.DataFrame({
            "Metric": ["Total Missions/Day", "Successful/Day", "Failed/Day", "On-Time Rate (%)", "Avg Response (min)",
                       "Avg Service (min)"],
            "Value": [round(tot, 1), round(suc, 1), round(fail, 1), f"{rate:.2f}%",
                      f"{df_resp['response_time_min'].mean():.2f}", f"{df_resp['service_duration_min'].mean():.2f}"]
        }).to_csv(SCRIPT_DIR / "KPI_summary_2.csv", index=False)

    if not df_unc.empty:
        stats = df_unc.groupby('node').agg(Frequency=('node', 'count'), Average_Duration_Min=('duration_min', 'mean'),
                                           Total_Duration_Min=('duration_min', 'sum')).reset_index()
        stats['Frequency'] = stats['Frequency'] / NUM_SEEDS
        stats['Total_Duration_Min'] = stats['Total_Duration_Min'] / NUM_SEEDS
        stats.round(2).to_csv(SCRIPT_DIR / "Node_Coverage_Stats_2.csv", index=False)

    if not df_resp.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(df_resp['response_time_min'], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(8, color='red', linestyle='dashed', linewidth=2, label='Severe (8m)')
        plt.axvline(12, color='orange', linestyle='dashed', linewidth=2, label='Mild (12m)')
        plt.title('Distribution of Response Times (Traffic-Aware - 30 Runs)', fontsize=14)
        plt.xlabel('Response Time (minutes)', fontsize=12);
        plt.ylabel('Count (Over 30 Days)', fontsize=12);
        plt.legend()
        plt.savefig(SCRIPT_DIR / "chart_response_times_2.png", dpi=150, bbox_inches='tight');
        plt.close()

    if not df_cov.empty:
        avg_cov = df_cov.groupby('minute')['uncovered_count'].mean().reset_index()
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(avg_cov)), avg_cov['uncovered_count'], color='brown', linewidth=2)
        plt.title('Average Uncovered Nodes over 24 Hours (30 Runs)', fontsize=14)
        plt.xlabel('Simulation Minute (0 = 06:00)', fontsize=12);
        plt.ylabel('Mean Uncovered Nodes', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(SCRIPT_DIR / "chart_uncovered_nodes_2.png", dpi=150, bbox_inches='tight');
        plt.close()


def main():
    nodes = load_nodes(SCRIPT_DIR / "nodes.csv")
    edges = load_edges(OUT_EDGES_PATH)
    t_col = next((c for c in edges.columns if 'traffic' in c.lower()), None)
    edges['base_traffic'] = edges[t_col] if t_col else 0.0

    all_inc, all_cov, all_resp, all_unc, all_out = [], [], [], [], []

    print(f"Running 30 iterations for Baseline Scenario...")
    for seed in range(1, NUM_SEEDS + 1):
        print(f"-> Simulating Day {seed}/{NUM_SEEDS}...")
        rng = np.random.default_rng(seed)
        G = build_graph(edges.copy(), nodes.copy())
        pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}
        ambs = create_ambulances(G)

        inc, cov, resp, evts, frames, unc, out = run_simulation(G, pos, ambs, rng, record_history=False)

        for r in inc: r['seed'] = seed; all_inc.append(r)
        for r in cov: r['seed'] = seed; all_cov.append(r)
        for r in resp: r['seed'] = seed; all_resp.append(r)
        for r in unc: r['seed'] = seed; all_unc.append(r)
        for r in out: r['seed'] = seed; all_out.append(r)

    pd.DataFrame(all_inc)[
        ["seed", "id", "node", "created_clock", "severity", "status", "arrival_clock", "service_end_clock"]].to_csv(
        INCIDENT_LOG, index=False)
    pd.DataFrame(all_cov).to_csv(COVERAGE_LOG, index=False)
    pd.DataFrame(all_resp).to_csv(SUMMARY_LOG, index=False)
    pd.DataFrame(all_out).to_csv(OUT_OF_SERVICE_LOG, index=False)

    generate_reports(pd.DataFrame(all_resp), pd.DataFrame(all_cov), pd.DataFrame(all_unc))
    print("\n✅ Simulation and Statistical Reports for Baseline completed successfully!")


if __name__ == "__main__":
    main()
