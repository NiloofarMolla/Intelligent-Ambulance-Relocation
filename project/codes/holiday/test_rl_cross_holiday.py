# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from tehran_graph import (
    SCRIPT_DIR, load_nodes, load_edges, build_graph, create_ambulances, STATION_NODES
)

OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_traffic.csv"
MODEL_PATH = SCRIPT_DIR / "ppo_ambulance_model"

INCIDENT_LOG = SCRIPT_DIR / "incident_log_cross_holiday.csv"
COVERAGE_LOG = SCRIPT_DIR / "coverage_log_cross_holiday.csv"
SUMMARY_LOG = SCRIPT_DIR / "response_summary_cross_holiday.csv"
OUT_OF_SERVICE_LOG = SCRIPT_DIR / "out_of_service_log_cross_holiday.csv"
REWARD_LOG = SCRIPT_DIR / "rl_reward_log_cross_holiday.csv"

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

W_MILD, W_SEVERE = 1.0, 10.0
LAMBDA_ON_TIME, LAMBDA_MISSED, LAMBDA_DELAY_RATIO = +100.0, -100.0, -50.0
LAMBDA_UNCOVERED, LAMBDA_RELOCATE, LAMBDA_UNSERVED, LAMBDA_WAITING_PER_MIN = -10.0, -0.15, -500.0, -2.0


def fmt_clock(m): return f"{m // 60:02d}:{m % 60:02d}" if m else ""


# 🌟 ترافیک تعطیلات
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
    m = get_time_multiplier(hour)
    for u, v, d in G.edges(data=True):
        C = float(d.get('base_traffic', 0.0)) * m
        if C < 0.4:
            d['current_time_min'] = d['base_time_min'] * (1.0 + C)
        elif C < 0.6:
            d['current_time_min'] = d['base_time_min'] * (2.0 + C)
        else:
            d['current_time_min'] = d['base_time_min'] * (3.0 + C)
    return dict(nx.all_pairs_dijkstra_path_length(G, weight="current_time_min")), dict(
        nx.all_pairs_dijkstra_path(G, weight="current_time_min"))


def plan_incidents(G, rng):
    sch = {t: [] for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES)}
    el = [n for n in G.nodes if n not in STATION_NODES]
    w = {n: (HOTSPOT_MULT if n in HOTSPOT_NODES else 1.0) for n in el}
    tw = sum(w.values());
    n_id = 1
    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        h = (t // 60) % 24
        rt = next(r for (hs, he), r in RATES_PER_HOUR.items() if hs <= h < he) / 60.0
        for n in el:
            for _ in range(rng.poisson(rt * (w[n] / tw))):
                sev = "severe" if rng.random() < 0.55 else "mild"
                sch[t].append({"id": f"INC-{n_id:04d}", "node": n, "created_min": t, "created_clock": fmt_clock(t),
                               "severity": sev, "deadline_min": 8.0 if sev == "severe" else 12.0, "status": "waiting",
                               "assigned": None, "arrival_min": None, "arrival_clock": None, "service_end_min": None,
                               "white_at": None})
                n_id += 1
    return sch


def assign_mission(amb, inc, t, k, times, paths):
    tgt = inc["node"]
    if tgt not in paths[amb["current_node"]]: return False
    amb.update({"route": paths[amb["current_node"]][tgt], "route_index": 0, "edge_elapsed": 0.0,
                "target_node": tgt, "incident_id": inc["id"] if k == "dispatch" else None,
                "status": "busy" if k == "dispatch" else "relocating", "_arrived_handled": False})
    if k == "dispatch": inc.update({"status": "assigned", "assigned": amb["id"]})
    return True


def step_movement(G, ambulances):
    for a in ambulances:
        if a["status"] not in ("busy", "relocating"): continue
        rem = 1.0
        while rem > 1e-9 and a["route_index"] < len(a["route"]) - 1:
            u, v = a["route"][a["route_index"]], a["route"][a["route_index"] + 1]
            left = G[u][v]["current_time_min"] - a["edge_elapsed"]
            if rem < left:
                a["edge_elapsed"] += rem; rem = 0.0
            else:
                rem -= left; a["route_index"] += 1; a["edge_elapsed"] = 0.0; a["current_node"] = v
        if a["route_index"] >= len(a["route"]) - 1:
            a.update({"current_node": a["route"][-1], "route": [a["route"][-1]], "route_index": 0, "edge_elapsed": 0.0})
            if not a.get("_arrived_handled"):
                a["_arrived_handled"] = True
                if a["status"] == "busy":
                    yield ("arrived", a)
                else:
                    a.update({"status": "free", "target_node": None}); yield ("relocated", a)


def get_coverage_states(G, ambulances, times):
    fr = [a["current_node"] for a in ambulances if a["status"] == "free"]
    ef = fr + [a["target_node"] for a in ambulances if a["status"] == "relocating" and a.get("target_node")]
    ac, log = set(), set()
    for n in G.nodes:
        if n in STATION_NODES: continue
        if min((times[p].get(n, np.inf) for p in fr), default=np.inf) > COVERAGE_LIMIT: ac.add(n)
        if min((times[p].get(n, np.inf) for p in ef), default=np.inf) > COVERAGE_LIMIT: log.add(n)
    return ac, log


def get_rl_obs(t, num_nodes, node_list, ambulances, times, last_multiplier):
    obs = np.zeros(3 + num_nodes, dtype=np.float32)
    obs[0], obs[1], obs[2] = (t - SIM_START_MIN) / SIM_MINUTES, last_multiplier, sum(
        1 for a in ambulances if a["status"] == "free") / max(1, len(ambulances))
    ef = [a["current_node"] for a in ambulances if a["status"] == "free"] + [a["target_node"] for a in ambulances if
                                                                             a["status"] == "relocating" and a.get(
                                                                                 "target_node")]
    for idx, node in enumerate(node_list):
        if node in STATION_NODES:
            obs[3 + idx] = 1.0
        else:
            obs[3 + idx] = 1.0 if min((times[p].get(node, np.inf) for p in ef),
                                      default=np.inf) <= COVERAGE_LIMIT else 0.0
    return obs


def run_rl_simulation(G, pos, ambulances, rng, model):
    node_list, num_nodes = list(G.nodes), G.number_of_nodes()
    incidents, sch = [], plan_incidents(G, rng)
    cov_rows, resp_rows, out_rows, rew_rows, unc_int, unc_since = [], [], [], [], [], {}
    times, paths, last_mult, cum_rew = None, None, -1, 0.0

    for t in range(SIM_START_MIN, SIM_START_MIN + SIM_MINUTES):
        clock, h = fmt_clock(t % (24 * 60)), (t // 60) % 24
        step_rew = 0.0
        cm = get_time_multiplier(h)
        if cm != last_mult:
            times, paths = update_traffic_and_paths(G, h)
            last_mult = cm

        for a in ambulances:
            if a["status"] == "out_of_service" and t >= a.get("out_of_service_until", 0):
                a.update({"status": "free", "out_of_service_until": None})

        if t % (24 * 60) == 0:
            for a in ambulances:
                if a["status"] == "free" and a["current_node"] != a["home_station"]:
                    if assign_mission(a, {"node": a["home_station"]}, t, "relocate", times,
                                      paths): step_rew += LAMBDA_RELOCATE

        for a in ambulances:
            if a["status"] == "free" and rng.random() < P_BREAKDOWN_PER_MIN:
                dt = int(rng.integers(OUT_OF_SERVICE_TIME_MIN[0], OUT_OF_SERVICE_TIME_MIN[1] + 1))
                a.update({"status": "out_of_service", "out_of_service_until": t + dt})
                out_rows.append(
                    {"ambulance_id": a["id"], "location_node": a["current_node"], "start_min": t, "start_clock": clock,
                     "duration_min": dt, "end_min": t + dt, "end_clock": fmt_clock(t + dt), "reason": "Breakdown"})

        for inc in sch.get(t, []): incidents.append(inc)
        waiting = sorted([i for i in incidents if i["status"] == "waiting"],
                         key=lambda x: (x["severity"] != "severe", x["created_min"]))

        def handle_arr(atm, inc_dict):
            nonlocal step_rew
            rt = atm - inc_dict["created_min"]
            w = W_SEVERE if inc_dict["severity"] == "severe" else W_MILD
            if rt <= inc_dict["deadline_min"]:
                step_rew += LAMBDA_ON_TIME * w
            else:
                step_rew += (LAMBDA_MISSED * w) + (
                            LAMBDA_DELAY_RATIO * w * (max(0, rt - inc_dict["deadline_min"]) / inc_dict["deadline_min"]))
            return rt, rt <= inc_dict["deadline_min"]

        for inc in waiting:
            fr = [a for a in ambulances if a["status"] == "free"]
            if not fr: break
            best, b_time = min(((a, times[a["current_node"]].get(inc["node"], np.inf)) for a in fr), key=lambda x: x[1],
                               default=(None, np.inf))
            if best is None or b_time == np.inf: continue

            if best["current_node"] == inc["node"]:
                best.update({"status": "busy", "route": [inc["node"]], "route_index": 0, "edge_elapsed": 0.0,
                             "target_node": inc["node"], "incident_id": inc["id"], "_arrived_handled": True})
                inc.update({"status": "arrived_local", "assigned": best["id"], "arrival_min": t, "arrival_clock": clock,
                            "white_at": t + 2})
                rt, ot = handle_arr(t, inc)
                sd = int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
                best["service_end_min"] = t + 2 + sd
                resp_rows.append({"incident_id": inc["id"], "node": inc["node"], "severity": inc["severity"],
                                  "created_clock": inc["created_clock"], "arrival_clock": inc["arrival_clock"],
                                  "response_time_min": rt, "deadline_min": inc["deadline_min"],
                                  "service_duration_min": sd, "on_time": ot, "ambulance": best["id"]})
            else:
                assign_mission(best, inc, t, "dispatch", times, paths)

        for inc in incidents:
            if inc["status"] == "arrived_local" and t >= inc["white_at"]: inc["status"] = "serving"

        for ev, amb in step_movement(G, ambulances):
            if ev == "arrived":
                inc = next(i for i in incidents if i["id"] == amb["incident_id"])
                inc.update({"arrival_min": t + 1, "arrival_clock": fmt_clock(t + 1), "status": "serving"})
                rt, ot = handle_arr(t + 1, inc)
                sd = int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
                amb["service_end_min"] = t + 1 + sd
                resp_rows.append({"incident_id": inc["id"], "node": inc["node"], "severity": inc["severity"],
                                  "created_clock": inc["created_clock"], "arrival_clock": inc["arrival_clock"],
                                  "response_time_min": rt, "deadline_min": inc["deadline_min"],
                                  "service_duration_min": sd, "on_time": ot, "ambulance": amb["id"]})

        for a in ambulances:
            if a["status"] == "busy" and a.get("service_end_min") is not None and t >= a["service_end_min"]:
                inc = next(i for i in incidents if i["id"] == a["incident_id"])
                inc.update({"status": "done", "service_end_min": t, "service_end_clock": clock})
                a.update({"service_end_min": None, "incident_id": None})
                if 0 <= h < 6 and a["current_node"] != a["home_station"]:
                    if assign_mission(a, {"node": a["home_station"]}, t, "relocate", times,
                                      paths): step_rew += LAMBDA_RELOCATE
                else:
                    a["status"] = "free"

        act_bad, log_bad = get_coverage_states(G, ambulances, times)
        step_rew += (len(act_bad) * LAMBDA_UNCOVERED)
        step_rew += sum(
            LAMBDA_WAITING_PER_MIN * (W_SEVERE if i["severity"] == "severe" else W_MILD) for i in incidents if
            i["status"] in ("waiting", "assigned", "arrived_local"))

        if 6 <= h < 24:
            obs = get_rl_obs(t, num_nodes, node_list, ambulances, times, last_mult)
            action, _ = model.predict(obs, deterministic=True)
            if action < num_nodes:
                tgt = node_list[action]
                fr = [a for a in ambulances if a["status"] == "free"]
                if fr:
                    best_a, b_t = min(((a, times[a["current_node"]].get(tgt, np.inf)) for a in fr), key=lambda x: x[1],
                                      default=(None, np.inf))
                    if best_a and best_a["current_node"] != tgt:
                        if assign_mission(best_a, {"node": tgt}, t, "relocate", times,
                                          paths): step_rew += LAMBDA_RELOCATE

        cum_rew += step_rew
        rew_rows.append(
            {"minute": t, "clock": clock, "step_reward": round(step_rew, 2), "cumulative_reward": round(cum_rew, 2)})

        for n in list(unc_since):
            if n not in act_bad:
                st = unc_since.pop(n)
                if t - st > 0: unc_int.append({"node": n, "start_min": st, "end_min": t, "duration_min": t - st})
        for n in act_bad:
            if n not in unc_since: unc_since[n] = t

        cov_rows.append({"minute": t, "clock": clock, "uncovered_count": len(act_bad),
                         "free": sum(a["status"] == "free" for a in ambulances),
                         "busy": sum(a["status"] == "busy" for a in ambulances),
                         "relocating": sum(a["status"] == "relocating" for a in ambulances),
                         "out_of_service": sum(a["status"] == "out_of_service" for a in ambulances)})

    cum_rew += sum(1 for i in incidents if i["status"] == "waiting") * LAMBDA_UNSERVED
    rew_rows[-1]["cumulative_reward"] = round(cum_rew, 2)
    for n, st in unc_since.items():
        if SIM_START_MIN + SIM_MINUTES - st > 0: unc_int.append(
            {"node": n, "start_min": st, "end_min": SIM_START_MIN + SIM_MINUTES,
             "duration_min": SIM_START_MIN + SIM_MINUTES - st})

    return incidents, cov_rows, resp_rows, unc_int, out_rows, rew_rows


def generate_reports(df_resp, df_cov, df_unc, df_rew):
    print("\n--- Generating RL Evaluation Reports (Holiday - 30-Day Avg) ---")
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
        }).to_csv(SCRIPT_DIR / "KPI_summary_cross_holiday.csv", index=False)

    if not df_unc.empty:
        stats = df_unc.groupby('node').agg(Frequency=('node', 'count'), Average_Duration_Min=('duration_min', 'mean'),
                                           Total_Duration_Min=('duration_min', 'sum')).reset_index()
        stats['Frequency'] = stats['Frequency'] / NUM_SEEDS
        stats['Total_Duration_Min'] = stats['Total_Duration_Min'] / NUM_SEEDS
        stats.round(2).to_csv(SCRIPT_DIR / "Node_Coverage_Stats_cross_holiday.csv", index=False)

    if not df_resp.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(df_resp['response_time_min'], bins=30, color='darkorange', edgecolor='black')
        plt.axvline(8, color='red', linestyle='dashed', lw=2, label='Severe (8m)')
        plt.axvline(12, color='blue', linestyle='dashed', lw=2, label='Mild (12m)')
        plt.title('Response Times (RL on Holiday - 30 Runs)', fontsize=14)
        plt.xlabel('Response Time (minutes)', fontsize=12);
        plt.ylabel('Count (Over 30 Days)', fontsize=12);
        plt.legend()
        plt.savefig(SCRIPT_DIR / "chart_response_times_cross_holiday.png", dpi=150, bbox_inches='tight');
        plt.close()

    if not df_cov.empty:
        avg_cov = df_cov.groupby('minute')['uncovered_count'].mean().reset_index()
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(avg_cov)), avg_cov['uncovered_count'], color='brown', lw=2)
        plt.title('Average Uncovered Nodes (RL on Holiday - 30 Runs)', fontsize=14)
        plt.xlabel('Simulation Minute (0 = 06:00)', fontsize=12);
        plt.ylabel('Mean Uncovered Nodes', fontsize=12)
        plt.grid(True, ls=':', alpha=0.6)
        plt.savefig(SCRIPT_DIR / "chart_uncovered_nodes_cross_holiday.png", dpi=150, bbox_inches='tight');
        plt.close()

    if not df_rew.empty:
        avg_rew = df_rew.groupby('minute')['cumulative_reward'].mean().reset_index()
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(avg_rew)), avg_rew['cumulative_reward'], color='purple', lw=2)
        plt.title('Average RL Cumulative Reward (Holiday - 30 Runs)', fontsize=14)
        plt.grid(True, ls=':', alpha=0.6);
        plt.axhline(0, color='black', lw=1)
        plt.fill_between(range(len(avg_rew)), avg_rew['cumulative_reward'], 0, where=(avg_rew['cumulative_reward'] < 0),
                         color='red', alpha=0.1)
        plt.fill_between(range(len(avg_rew)), avg_rew['cumulative_reward'], 0,
                         where=(avg_rew['cumulative_reward'] >= 0), color='green', alpha=0.1)
        plt.savefig(SCRIPT_DIR / "chart_rl_rewards_eval_cross_holiday.png", dpi=150, bbox_inches='tight');
        plt.close()


def main():
    nodes = load_nodes(SCRIPT_DIR / "nodes.csv")
    edges = load_edges(OUT_EDGES_PATH)
    t_col = next((c for c in edges.columns if 'traffic' in c.lower()), None)
    edges['base_traffic'] = edges[t_col] if t_col else 0.0

    G = build_graph(edges, nodes)
    pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        fail(f"Model load error: {e}")

    all_inc, all_cov, all_resp, all_unc, all_out, all_rew = [], [], [], [], [], []

    print(f"Running 30 iterations for RL Agent (CROSS-EVAL HOLIDAY)...")
    for seed in range(1, NUM_SEEDS + 1):
        print(f"-> Simulating Day {seed}/{NUM_SEEDS}...")
        rng = np.random.default_rng(seed)
        G_sim = build_graph(edges.copy(), nodes.copy())
        ambs = create_ambulances(G_sim)

        inc, cov, resp, unc, out, rew = run_rl_simulation(G_sim, pos, ambs, rng, model)

        for r in inc: r['seed'] = seed; all_inc.append(r)
        for r in cov: r['seed'] = seed; all_cov.append(r)
        for r in resp: r['seed'] = seed; all_resp.append(r)
        for r in unc: r['seed'] = seed; all_unc.append(r)
        for r in out: r['seed'] = seed; all_out.append(r)
        for r in rew: r['seed'] = seed; all_rew.append(r)

    pd.DataFrame(all_inc)[
        ["seed", "id", "node", "created_clock", "severity", "status", "arrival_clock", "service_end_clock"]].to_csv(
        INCIDENT_LOG, index=False)
    pd.DataFrame(all_cov).to_csv(COVERAGE_LOG, index=False)
    pd.DataFrame(all_resp).to_csv(SUMMARY_LOG, index=False)
    pd.DataFrame(all_out).to_csv(OUT_OF_SERVICE_LOG, index=False)
    pd.DataFrame(all_rew).to_csv(REWARD_LOG, index=False)

    generate_reports(pd.DataFrame(all_resp), pd.DataFrame(all_cov), pd.DataFrame(all_unc), pd.DataFrame(all_rew))
    print("\n✅ Simulation and Statistical Reports for RL Agent (Holiday) completed successfully!")


if __name__ == "__main__":
    main()