# -*- coding: utf-8 -*-
"""
RL Agent Training Phase for Tehran Ambulance Dispatch.
Uses Gymnasium and Stable-Baselines3 (PPO Algorithm).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from tehran_graph import (
    SCRIPT_DIR, load_nodes, load_edges, build_graph, create_ambulances, STATION_NODES
)

# ----------------------------------------------------------------------------
# Configurations & RL Hyperparameters
# ----------------------------------------------------------------------------
OUT_EDGES_PATH = SCRIPT_DIR / "edges_with_traffic.csv"
MODEL_SAVE_PATH = SCRIPT_DIR / "ppo_ambulance_model"

SIM_START_MIN = 6 * 60
SIM_MINUTES = 24 * 60
COVERAGE_LIMIT = 8.0
SERVICE_TIME_MIN = (25, 40)

# <--- متغیر جا افتاده اضافه شد --->
RATES_PER_HOUR = {
    (0, 6): 6,
    (6, 9): 25,
    (9, 12): 15,
    (12, 16): 15,
    (16, 20): 25,
    (20, 24): 22,
}

# Reward Weights
W_MILD, W_SEVERE = 1.0, 10.0
LAMBDA_ON_TIME = +100.0
LAMBDA_MISSED = -100.0
LAMBDA_DELAY_RATIO = -50.0
LAMBDA_UNCOVERED = -10.0
LAMBDA_RELOCATE = -0.15
LAMBDA_UNSERVED = -500.0
LAMBDA_WAITING_PER_MIN = -2.0


# ----------------------------------------------------------------------------
# Traffic & Path Helpers
# ----------------------------------------------------------------------------
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
            data['current_time_min'] = data['base_time_min'] * (1.0 + C)
        elif C < 0.6:
            data['current_time_min'] = data['base_time_min'] * (2.0 + C)
        else:
            data['current_time_min'] = data['base_time_min'] * (3.0 + C)
    times = dict(nx.all_pairs_dijkstra_path_length(G, weight="current_time_min"))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight="current_time_min"))
    return times, paths


# ----------------------------------------------------------------------------
# Custom Gymnasium Environment
# ----------------------------------------------------------------------------
class TehranAmbulanceEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Agent learns to relocate ambulances to minimize response time and maximize coverage.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(TehranAmbulanceEnv, self).__init__()

        self.nodes_df = load_nodes(SCRIPT_DIR / "nodes.csv")
        self.edges_df = load_edges(OUT_EDGES_PATH)

        traffic_col = next((c for c in self.edges_df.columns if 'traffic' in c.lower()), None)
        self.edges_df['base_traffic'] = self.edges_df[traffic_col] if traffic_col else 0.0

        self.G = build_graph(self.edges_df, self.nodes_df)
        self.num_nodes = self.G.number_of_nodes()
        self.node_list = list(self.G.nodes)

        self.action_space = spaces.Discrete(self.num_nodes + 1)

        # 🌟 تغییر مهم: سایز مشاهدات را از 1 به 3 افزایش دادیم
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 + self.num_nodes,), dtype=np.float32
        )
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.t = SIM_START_MIN
        self.ambulances = create_ambulances(self.G)
        self.incidents = []

        self.times, self.paths = update_traffic_and_paths(self.G, (self.t // 60) % 24)
        self.last_multiplier = get_time_multiplier((self.t // 60) % 24)

        self.cumulative_reward = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        """ Returns the current state of the environment as a numpy array. """
        obs = np.zeros(3 + self.num_nodes, dtype=np.float32)

        # 1. زمان نرمال شده
        obs[0] = (self.t - SIM_START_MIN) / SIM_MINUTES
        # 2. ضریب ترافیک فعلی (تا هوش مصنوعی پیک ترافیک را حس کند)
        obs[1] = self.last_multiplier
        # 3. نسبت آمبولانس‌های آزاد (تا بفهمد چقدر نیرو برایش مانده)
        obs[2] = sum(1 for a in self.ambulances if a["status"] == "free") / max(1, len(self.ambulances))

        effective_positions = [a["current_node"] for a in self.ambulances if a["status"] == "free"]
        for a in self.ambulances:
            if a["status"] == "relocating" and a.get("target_node"):
                effective_positions.append(a["target_node"])

        for idx, node in enumerate(self.node_list):
            if node in STATION_NODES:
                obs[3 + idx] = 1.0
            else:
                best = min((self.times[p].get(node, np.inf) for p in effective_positions), default=np.inf)
                obs[3 + idx] = 1.0 if best <= COVERAGE_LIMIT else 0.0

        return obs

    def step(self, action):
        step_reward = 0.0
        hour = (self.t // 60) % 24

        # 1. Update Traffic
        current_multiplier = get_time_multiplier(hour)
        if current_multiplier != self.last_multiplier:
            self.times, self.paths = update_traffic_and_paths(self.G, hour)
            self.last_multiplier = current_multiplier

        # 2. Night Sweep
        if self.t % (24 * 60) == 0:
            for amb in self.ambulances:
                if amb["status"] == "free" and amb["current_node"] != amb["home_station"]:
                    self._assign_mission(amb, amb["home_station"], "relocate")
                    step_reward += LAMBDA_RELOCATE

        # 3. Generate New Incidents
        city_rate = RATES_PER_HOUR.get(next((k for k in RATES_PER_HOUR if k[0] <= hour < k[1]), (9, 12)), 10) / 60.0
        if self.rng.random() < city_rate:
            node = self.rng.choice([n for n in self.node_list if n not in STATION_NODES])
            severity = "severe" if self.rng.random() < 0.35 else "mild"
            deadline = 8.0 if severity == "severe" else 12.0
            self.incidents.append({
                "id": f"INC-{self.t}", "node": node, "created_min": self.t,
                "severity": severity, "deadline_min": deadline, "status": "waiting",
                "assigned": None, "arrival_min": None, "white_at": None, "service_end_min": None
            })

        # 4. Dispatch Greedily (Rule-based)
        waiting = [i for i in self.incidents if i["status"] == "waiting"]
        waiting.sort(key=lambda i: (i["severity"] != "severe", i["created_min"]))

        for inc in waiting:
            free = [a for a in self.ambulances if a["status"] == "free"]
            if not free: break
            best_amb, best_time = min(
                ((a, self.times[a["current_node"]].get(inc["node"], np.inf)) for a in free),
                key=lambda x: x[1], default=(None, np.inf)
            )

            if best_amb and best_time != np.inf:
                if best_amb["current_node"] == inc["node"]:
                    best_amb.update({"status": "busy", "route": [inc["node"]], "route_index": 0, "edge_elapsed": 0.0,
                                     "target_node": inc["node"], "incident_id": inc["id"]})
                    inc.update({"status": "serving", "assigned": best_amb["id"], "arrival_min": self.t})
                    rt = self.t - inc["created_min"]
                    step_reward += self._calc_arrival_reward(rt, inc)
                    best_amb["service_end_min"] = self.t + int(self.rng.integers(*SERVICE_TIME_MIN))
                else:
                    self._assign_mission(best_amb, inc["node"], "dispatch", inc)

        # 5. Apply RL Action (Relocation)
        if 6 <= hour < 24 and action < self.num_nodes:
            target_node = self.node_list[action]

            # Agent wants to relocate an ambulance to target_node
            free_ambs = [a for a in self.ambulances if a["status"] == "free"]
            if free_ambs:
                # Pick closest safe ambulance
                best_amb, best_time = min(
                    ((a, self.times[a["current_node"]].get(target_node, np.inf)) for a in free_ambs),
                    key=lambda x: x[1], default=(None, np.inf)
                )
                if best_amb and best_amb["current_node"] != target_node:
                    self._assign_mission(best_amb, target_node, "relocate")
                    step_reward += LAMBDA_RELOCATE  # Penalty for taking action

        # 6. Step Ambulances Movement
        for amb in self.ambulances:
            if amb["status"] not in ("busy", "relocating"): continue

            remaining = 1.0
            while remaining > 1e-9 and amb["route_index"] < len(amb["route"]) - 1:
                u, v = amb["route"][amb["route_index"]], amb["route"][amb["route_index"] + 1]
                left = self.G[u][v]["current_time_min"] - amb["edge_elapsed"]
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
                amb["route_index"], amb["edge_elapsed"] = 0, 0.0

                if amb["status"] == "busy":
                    inc = next((i for i in self.incidents if i["id"] == amb["incident_id"]), None)
                    if inc:
                        inc["arrival_min"] = self.t
                        inc["status"] = "serving"
                        rt = self.t - inc["created_min"]
                        step_reward += self._calc_arrival_reward(rt, inc)
                        amb["service_end_min"] = self.t + int(self.rng.integers(*SERVICE_TIME_MIN))
                else:
                    amb["status"], amb["target_node"] = "free", None

        # 7. Release Ambulances
        for amb in self.ambulances:
            if amb["status"] == "busy" and amb.get("service_end_min") is not None and self.t >= amb["service_end_min"]:
                inc = next((i for i in self.incidents if i["id"] == amb["incident_id"]), None)
                if inc: inc["status"] = "done"
                amb.update({"status": "free", "service_end_min": None, "incident_id": None})

        # 8. Calculate Penalties
        obs = self._get_obs()
        uncovered_count = list(obs[1:]).count(0.0)
        step_reward += (uncovered_count * LAMBDA_UNCOVERED)

        for inc in self.incidents:
            if inc["status"] in ("waiting", "assigned"):
                w = W_SEVERE if inc["severity"] == "severe" else W_MILD
                step_reward += LAMBDA_WAITING_PER_MIN * w

        self.t += 1
        self.cumulative_reward += step_reward
        done = bool(self.t >= SIM_START_MIN + SIM_MINUTES)

        # End of episode penalty
        if done:
            unserved = sum(1 for i in self.incidents if i["status"] == "waiting")
            step_reward += unserved * LAMBDA_UNSERVED

        return obs, step_reward, done, False, {}

    def _calc_arrival_reward(self, rt, inc):
        w = W_SEVERE if inc["severity"] == "severe" else W_MILD
        if rt <= inc["deadline_min"]:
            return LAMBDA_ON_TIME * w
        else:
            delay_ratio = max(0, rt - inc["deadline_min"]) / inc["deadline_min"]
            return (LAMBDA_MISSED * w) + (LAMBDA_DELAY_RATIO * w * delay_ratio)

    def _assign_mission(self, amb, target, kind, incident=None):
        if target not in self.paths[amb["current_node"]]: return False
        amb.update({
            "route": self.paths[amb["current_node"]][target], "route_index": 0, "edge_elapsed": 0.0,
            "target_node": target, "incident_id": incident["id"] if incident else None,
            "status": "busy" if kind == "dispatch" else "relocating"
        })
        if incident:
            incident.update({"status": "assigned", "assigned": amb["id"]})
        return True


# ----------------------------------------------------------------------------
# Main Training Loop
# ----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 50)
    print("INITIALIZING REINFORCEMENT LEARNING ENVIRONMENT")
    print("Algorithm: Proximal Policy Optimization (PPO)")
    print("=" * 50)

    # ساخت محیط
    env = TehranAmbulanceEnv()

    # ساخت مدل PPO (شبکه عصبی)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)

    print("\n--- Starting Training Process ---")
    print("This may take 10 to 30 minutes depending on your CPU.")
    print("The agent is exploring the city, making mistakes, and learning...")

    # آموزش مدل برای 50,000 قدم
    TOTAL_TIMESTEPS = 1500000

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # ذخیره مغز آموزش‌دیده
    model.save(MODEL_SAVE_PATH)
    print(f"\nSUCCESS! Model successfully trained and saved to {MODEL_SAVE_PATH}.zip")
    print("You can now load this model to evaluate its intelligent decisions.")


if __name__ == "__main__":
    main()