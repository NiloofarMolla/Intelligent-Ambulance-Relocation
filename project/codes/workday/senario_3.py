# -*- coding: utf-8 -*-
"""
Baseline vs RL Reward Comparison (30-Day Avg)
Executes the heuristic baseline under the MDP reward function to prove AI superiority.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path

from tehran_graph import (
    SCRIPT_DIR, load_nodes, load_edges, build_graph, create_ambulances, STATION_NODES
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_traffic.csv"
REWARD_LOG_BASELINE = SCRIPT_DIR / "rl_reward_log_baseline_30.csv"
REWARD_LOG_RL = SCRIPT_DIR / "rl_reward_log_eval.csv" 

NUM_SEEDS = 30
SIM_START_MIN = 6 * 60
SIM_MINUTES = 24 * 60
COVERAGE_LIMIT = 8.0
SERVICE_TIME_MIN = (25, 40)
P_BREAKDOWN_PER_MIN = 0.0005
OUT_OF_SERVICE_TIME_MIN = (30, 40)
HOTSPOT_NODES = {"n12", "n13", "n19", "n24", "n25", "n26", "n31", "n43", "n45"}
HOTSPOT_MULT = 1.8

RATES_PER_HOUR = {(0, 6): 6, (6, 9): 25, (9, 12): 15, (12, 16): 15, (16, 20): 25, (20, 24): 22}

# MDP Parameters
W_MILD, W_SEVERE = 1.0, 10.0
LAMBDA_ON_TIME = +100.0
LAMBDA_MISSED = -100.0
LAMBDA_DELAY_RATIO = -50.0
LAMBDA_UNCOVERED = -10.0
LAMBDA_RELOCATE = -0.15
LAMBDA_UNSERVED = -500.0
LAMBDA_WAITING_PER_MIN = -2.0


def fmt_clock(m): return f"{m // 60:02d}:{m % 60:02d}" if m else ""


def get_time_multiplier(h):
    if 0 <= h < 6: return 0.1
    if 6 <= h < 9: return 0.9
    if 9 <= h < 12: return 0.5
    if 12 <= h < 16: return 0.6
    if 16 <= h < 20: return 0.9
    return 0.4


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
                sev = "severe" if rng.random() < 0.35 else "mild"
                sch[t].append({"id": f"INC-{n_id:04d}", "node": n, "created_min": t, "created_clock": fmt_clock(t),
                               "severity": sev, "deadline_min": 8.0 if sev == "severe" else 12.0, "status": "waiting",
                               "assigned": None, "arrival_min": None, "arrival_clock": None, "service_end_min": None,
                               "white_at": None})
                n_id += 1
    return sch


def assign_mission(amb, inc, t, k, times, paths):
    tgt = inc["node"]
    if tgt not in paths[amb["current_node"]]: return False
    amb.update({"route": paths[amb["current_node"]][tgt], "route_index": 0, "edge_elapsed": 0.0, "target_node": tgt,
                "incident_id": inc["id"] if k == "dispatch" else None,
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


def run_baseline_with_rewards(G, pos, ambulances, rng):
    incidents, sch = [], plan_incidents(G, rng)
    rew_rows = []
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

        for inc in waiting:
            fr = [a for a in ambulances if a["status"] == "free"]
            if not fr: break
            best_amb, best_time = min(((a, times[a["current_node"]].get(inc["node"], np.inf)) for a in fr),
                                      key=lambda x: x[1], default=(None, np.inf))
            if best_amb is None or best_time == np.inf: continue

            if best_amb["current_node"] == inc["node"]:
                best_amb.update({"status": "busy", "route": [inc["node"]], "route_index": 0, "edge_elapsed": 0.0,
                                 "target_node": inc["node"], "incident_id": inc["id"], "_arrived_handled": True})
                inc.update(
                    {"status": "arrived_local", "assigned": best_amb["id"], "arrival_min": t, "arrival_clock": clock,
                     "white_at": t + 2})
                handle_arr(t, inc)
                best_amb["service_end_min"] = t + 2 + int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))
            else:
                assign_mission(best_amb, inc, t, "dispatch", times, paths)

        for inc in incidents:
            if inc["status"] == "arrived_local" and t >= inc["white_at"]: inc["status"] = "serving"

        for ev, amb in step_movement(G, ambulances):
            if ev == "arrived":
                inc = next(i for i in incidents if i["id"] == amb["incident_id"])
                inc.update({"arrival_min": t + 1, "arrival_clock": fmt_clock(t + 1), "status": "serving"})
                handle_arr(t + 1, inc)
                amb["service_end_min"] = t + 1 + int(rng.integers(SERVICE_TIME_MIN[0], SERVICE_TIME_MIN[1] + 1))

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

        # -------------------------------------------------------------------
        # BASELINE RELOCATION (Reactive) - No AI, just heuristic
        # -------------------------------------------------------------------
        act_bad, log_bad = get_coverage_states(G, ambulances, times)
        step_rew += (len(act_bad) * LAMBDA_UNCOVERED)
        step_rew += sum(
            LAMBDA_WAITING_PER_MIN * (W_SEVERE if i["severity"] == "severe" else W_MILD) for i in incidents if
            i["status"] in ("waiting", "assigned", "arrived_local"))

        if 6 <= h < 24 and log_bad:
            for tgt in list(log_bad):
                fr = [a for a in ambulances if a["status"] == "free"]
                if not fr: break
                best_a, b_t = None, np.inf
                for amb in fr:
                    d = times[amb["current_node"]].get(tgt, np.inf)
                    if d >= b_t: continue
                    oth = [a["current_node"] for a in ambulances if a is not amb and a["status"] == "free"] + [
                        a["target_node"] for a in ambulances if a is not amb and a["status"] == "relocating"]
                    safe = True
                    for n in G.nodes:
                        if n in STATION_NODES or n == tgt: continue
                        if times[amb["current_node"]].get(n, np.inf) <= COVERAGE_LIMIT:
                            if not oth or min((times[p].get(n, np.inf) for p in oth), default=np.inf) > COVERAGE_LIMIT:
                                if times[tgt].get(n, np.inf) > COVERAGE_LIMIT: safe = False; break
                    if safe: best_a, b_t = amb, d
                if best_a and best_a["current_node"] != tgt:
                    if assign_mission(best_a, {"node": tgt}, t, "relocate", times,
                                      paths): step_rew += LAMBDA_RELOCATE; log_bad.remove(tgt)

        cum_rew += step_rew
        rew_rows.append(
            {"minute": t, "clock": clock, "step_reward": round(step_rew, 2), "cumulative_reward": round(cum_rew, 2)})

    cum_rew += sum(1 for i in incidents if i["status"] == "waiting") * LAMBDA_UNSERVED
    rew_rows[-1]["cumulative_reward"] = round(cum_rew, 2)
    return rew_rows


def main():
    print("\n" + "=" * 60)
    print("CALCULATING MDP REWARDS FOR BASELINE (30-DAY AVG)")
    print("=" * 60)
    nodes = load_nodes(SCRIPT_DIR / "nodes.csv")
    edges = load_edges(OUT_EDGES_PATH)
    t_col = next((c for c in edges.columns if 'traffic' in c.lower()), None)
    edges['base_traffic'] = edges[t_col] if t_col else 0.0

    all_rew = []
    for seed in range(1, NUM_SEEDS + 1):
        print(f"-> Simulating Baseline Reward Day {seed}/{NUM_SEEDS}...")
        rng = np.random.default_rng(seed)
        G = build_graph(edges.copy(), nodes.copy())
        pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}
        ambs = create_ambulances(G)

        rew = run_baseline_with_rewards(G, pos, ambs, rng)
        for r in rew: r['seed'] = seed; all_rew.append(r)

    df_base = pd.DataFrame(all_rew)
    df_base.to_csv(REWARD_LOG_BASELINE, index=False)

    if not REWARD_LOG_RL.exists():
        print(f"Error: {REWARD_LOG_RL.name} not found! Cannot generate comparative plot.")
        return

    df_rl = pd.read_csv(REWARD_LOG_RL)

    avg_base = df_base.groupby('minute')['cumulative_reward'].mean().reset_index()
    avg_rl = df_rl.groupby('minute')['cumulative_reward'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(avg_base['minute'], avg_base['cumulative_reward'], color='red', label='Dynamic Baseline (No AI)',
             linewidth=2.5)
    plt.plot(avg_rl['minute'], avg_rl['cumulative_reward'], color='green', label='Trained Intelligent Agent',
             linewidth=2.5)

    plt.fill_between(avg_base['minute'], avg_base['cumulative_reward'], avg_rl['cumulative_reward'], color='lightgreen',
                     alpha=0.3, label='Penalties Prevented by AI')

    plt.axhline(0, color='black', linewidth=1)
    plt.title('Cost-Minimization Proof: RL Agent vs Baseline (30-Run Average)', fontsize=15, fontweight='bold')
    plt.xlabel('Simulation Minute (0 = 06:00)', fontsize=12)
    plt.ylabel('Cumulative Reward / Penalty', fontsize=12)
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = SCRIPT_DIR / "chart_comparison_rewards_30runs.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparative Reward Chart successfully saved as: {save_path.name}")
    plt.show()


if __name__ == "__main__":
    main()
