# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

from tehran_graph import (
    SCRIPT_DIR, load_nodes, load_edges, build_graph, create_ambulances, STATION_NODES
)

# Config for Scenario 2 (Holiday)
OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_traffic.csv"
INCIDENT_LOG = SCRIPT_DIR / "incident_log_2_holiday.csv"
COVERAGE_LOG = SCRIPT_DIR / "coverage_log_2_holiday.csv"
SUMMARY_LOG = SCRIPT_DIR / "response_summary_2_holiday.csv"
OUT_OF_SERVICE_LOG = SCRIPT_DIR / "out_of_service_log_2_holiday.csv"

NUM_SEEDS = 30  # 🌟 اجرای 30 روزه

SIM_START_MIN = 6 * 60
SIM_MINUTES = 24 * 60
COVERAGE_LIMIT = 8.0

SERVICE_TIME_MIN = (25, 40)
P_BREAKDOWN_PER_MIN = 0.0005
OUT_OF_SERVICE_TIME_MIN = (30, 40)
HOTSPOT_NODES = {"n12", "n13", "n19", "n24", "n25", "n26", "n31", "n43", "n45"}
HOTSPOT_MULT = 1.8

# 🌟 الگوهای تقاضای روز تعطیل
RATES_PER_HOUR = {
    (0, 7): 12, (7, 11): 8, (11, 14): 15, (14, 18): 12, (18, 22): 20, (22, 24): 12,
}


def fmt_clock(minute_of_day):
    if minute_of_day is None: return ""
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


# 🌟 الگوی ترافیک روز تعطیل
def get_time_multiplier(hour):
    if 7 <= hour < 11:
        return 0.15
    elif 11 <= hour < 14:
        return 0.55
    elif 14 <= hour < 18:
        return 0.40
    elif 18 <= hour < 22:
        return 0.80
    else:
        return 0.20


def update_traffic_and_paths(G, hour):
    mult = get_time_multiplier(hour)
    for u, v, data in G.edges(data=True):
        C = float(data.get('base_traffic', 0.0)) * mult
        if C < 0.4:
            data['current_time_min'] = data['base_time_min'] * (1.0 + C)
        elif C < 0.6:
            data['current_time_min'] = data['base_time_min'] * (2.0 + C)
        else:
            data['current_time_min'] = data['base_time_min'] * (3.0 + C)
    return dict(nx.all_pairs_dijkstra_path_length(G, weight="current_time_min")), dict(
        nx.all_pairs_dijkstra_path(G, weight="current_time_min"))


def plan_incidents(G, rng):
    schedule = {t: [] for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES)}
    eligible_nodes = [n for n in G.nodes if n not in STATION_NODES]
    weights = {n: (HOTSPOT_MULT if n in HOTSPOT_NODES else 1.0) for n in eligible_nodes}
    tot_weight = sum(weights.values());
    next_id = 1
    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        hour = (t // 60) % 24
        rate = next(r for (hs, he), r in RATES_PER_HOUR.items() if hs <= hour < he) / 60.0
        for node in eligible_nodes:
            count = rng.poisson(rate * (weights[node] / tot_weight))
            for _ in range(count):
                sev = "severe" if rng.random() < 0.55 else "mild"  # 55% severe for holiday
                schedule[t].append({
                    "id": f"INC-{next_id:04d}", "node": node, "created_min": t, "created_clock": fmt_clock(t),
                    "severity": sev, "deadline_min": 8.0 if sev == "severe" else 12.0, "status": "waiting",
                    "assigned": None, "arrival_min": None, "arrival_clock": None, "service_end_min": None,
                    "white_at": None
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


def run_simulation(G, pos, ambulances, rng):
    incidents = []
    incident_schedule = plan_incidents(G, rng)
    cov_rows, resp_rows, out_rows, uncov_intervals = [], [], [], []
    uncov_since = {}
    times, paths, last_mult = None, None, -1

    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        clock, hour = fmt_clock(t % (24 * 60)), (t // 60) % 24
        cur_mult = get_time_multiplier(hour)
        if cur_mult != last_mult:
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

        for inc in incident_schedule.get(t, []): incidents.append(inc)

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

    end_t = SIM_START_MIN + SIM_MINUTES
    for n, st in uncov_since.items():
        if end_t - st > 0: uncov_intervals.append(
            {"node": n, "start_min": st, "end_min": end_t, "duration_min": end_t - st})
    return incidents, cov_rows, resp_rows, uncov_intervals, out_rows


def generate_reports(df_resp, df_cov, df_unc):
    print("\n--- Generating Scenario 2 Reports (HOLIDAY - 30-Day Avg) ---")
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
        }).to_csv(SCRIPT_DIR / "KPI_summary_2_holiday.csv", index=False)

    if not df_unc.empty:
        stats = df_unc.groupby('node').agg(Frequency=('node', 'count'), Average_Duration_Min=('duration_min', 'mean'),
                                           Total_Duration_Min=('duration_min', 'sum')).reset_index()
        stats['Frequency'] = stats['Frequency'] / NUM_SEEDS
        stats['Total_Duration_Min'] = stats['Total_Duration_Min'] / NUM_SEEDS
        stats.round(2).to_csv(SCRIPT_DIR / "Node_Coverage_Stats_2_holiday.csv", index=False)

    if not df_resp.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(df_resp['response_time_min'], bins=30, color='skyblue', edgecolor='black')
        plt.axvline(8, color='red', linestyle='dashed', linewidth=2, label='Severe (8m)')
        plt.axvline(12, color='orange', linestyle='dashed', linewidth=2, label='Mild (12m)')
        plt.title('Distribution of Response Times (Baseline Holiday - 30 Runs)', fontsize=14)
        plt.xlabel('Response Time (minutes)', fontsize=12);
        plt.ylabel('Count (Over 30 Days)', fontsize=12);
        plt.legend()
        plt.savefig(SCRIPT_DIR / "chart_response_times_2_holiday.png", dpi=150, bbox_inches='tight');
        plt.close()

    if not df_cov.empty:
        avg_cov = df_cov.groupby('minute')['uncovered_count'].mean().reset_index()
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(avg_cov)), avg_cov['uncovered_count'], color='brown', linewidth=2)
        plt.title('Average Uncovered Nodes over 24 Hours (Baseline Holiday - 30 Runs)', fontsize=14)
        plt.xlabel('Simulation Minute (0 = 06:00)', fontsize=12);
        plt.ylabel('Mean Uncovered Nodes', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(SCRIPT_DIR / "chart_uncovered_nodes_2_holiday.png", dpi=150, bbox_inches='tight');
        plt.close()


def main():
    nodes = load_nodes(SCRIPT_DIR / "nodes.csv")
    edges = load_edges(OUT_EDGES_PATH)
    t_col = next((c for c in edges.columns if 'traffic' in c.lower()), None)
    edges['base_traffic'] = edges[t_col] if t_col else 0.0

    all_inc, all_cov, all_resp, all_unc, all_out = [], [], [], [], []

    print(f"Running 30 iterations for Baseline Scenario (HOLIDAY)...")
    for seed in range(1, NUM_SEEDS + 1):
        print(f"-> Simulating Day {seed}/{NUM_SEEDS}...")
        rng = np.random.default_rng(seed)
        G = build_graph(edges.copy(), nodes.copy())
        pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}
        ambs = create_ambulances(G)

        inc, cov, resp, unc, out = run_simulation(G, pos, ambs, rng)

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
    print("\n✅ Simulation and Statistical Reports for Baseline (HOLIDAY) completed successfully!")


if __name__ == "__main__":
    main()