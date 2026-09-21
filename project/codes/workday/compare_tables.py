import pandas as pd
import os
from pathlib import Path

SCRIPT_DIR = Path(os.getcwd())


def get_metrics(scenario_name, cov_file, node_cov_file, resp_file):
    try:
        df_cov = pd.read_csv(cov_file)
        df_node = pd.read_csv(node_cov_file)
        df_resp = pd.read_csv(resp_file)

        # استخراج تعداد روزها (Seeds)
        num_seeds = df_resp['seed'].nunique() if 'seed' in df_resp.columns else 1

        avg_uncov_nodes = round(df_cov['uncovered_count'].mean(), 2)
        avg_uncov_duration = round(df_node['Average_Duration_Min'].mean(), 2)

        # تقسیم مجموع جابجایی بر 30 روز
        avg_total_reloc = int(df_cov['relocating'].sum() / num_seeds)

        # بار کاری هر آمبولانس به طور میانگین در روز
        workload_per_seed = df_resp.groupby(['seed', 'ambulance']).size().reset_index(name='missions')
        avg_workload = workload_per_seed.groupby('ambulance')['missions'].mean()

        workload_std = round(avg_workload.std(), 2)
        max_workload = int(avg_workload.max())

        return {
            "Scenario": scenario_name,
            "Avg Uncovered Nodes (Count)": avg_uncov_nodes,
            "Avg Uncovered Duration (Min)": avg_uncov_duration,
            "Total Relocation Time / Day (Min)": avg_total_reloc,
            "Workload Std Dev (Missions)": workload_std,
            "Max Workload / Day (Missions)": max_workload
        }
    except Exception as e:
        print(f"Skipping '{scenario_name}': {e}")
        return None


def main():
    print("\n" + "=" * 60)
    print("GENERATING 30-DAY AVERAGED COMPARATIVE BENCHMARK TABLES")
    print("=" * 60)

    w_metrics = []
    m1 = get_metrics("Baseline (No AI)", SCRIPT_DIR / "coverage_log_2.csv", SCRIPT_DIR / "Node_Coverage_Stats_2.csv",
                     SCRIPT_DIR / "response_summary_2.csv")
    if m1: w_metrics.append(m1)
    m2 = get_metrics("RL Agent", SCRIPT_DIR / "coverage_log_rl.csv", SCRIPT_DIR / "Node_Coverage_Stats_rl.csv",
                     SCRIPT_DIR / "response_summary_rl.csv")
    if m2: w_metrics.append(m2)

    if w_metrics:
        df_w = pd.DataFrame(w_metrics)
        df_w.to_csv(SCRIPT_DIR / "Comparison_Table_Workday_30Avg.csv", index=False)
        print("\n--- Workday Comparison Table (Average of 30 Runs) ---")
        print(df_w.to_markdown(index=False))


if __name__ == "__main__":
    main()