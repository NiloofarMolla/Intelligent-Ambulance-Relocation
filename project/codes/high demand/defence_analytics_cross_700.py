import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

SCRIPT_DIR = Path(os.getcwd())
COVERAGE_FILE = SCRIPT_DIR / "Node_Coverage_Stats_cross_700.csv"
RESPONSE_FILE = SCRIPT_DIR / "response_summary_cross_700.csv"
NODES_FILE = SCRIPT_DIR / "nodes.csv"


def main():
    print("\n--- Generating Advanced Defense Analytics (High Demand 700 - 30-Day Avg) ---")
    node_mapping = {}
    if NODES_FILE.exists():
        df_nodes = pd.read_csv(NODES_FILE)
        df_nodes['name'] = df_nodes['name'].fillna(df_nodes['id'])
        node_mapping = dict(zip(df_nodes['id'], df_nodes['name']))

    def get_node_name(node_id):
        return node_mapping.get(node_id, str(node_id))

    if RESPONSE_FILE.exists():
        df_resp = pd.read_csv(RESPONSE_FILE)
        num_seeds = df_resp['seed'].nunique() if 'seed' in df_resp.columns else 1

        if COVERAGE_FILE.exists():
            df_cov = pd.read_csv(COVERAGE_FILE)
            df_cov['node_name'] = df_cov['node'].apply(get_node_name)
            df_cov_sorted = df_cov.sort_values(by='Total_Duration_Min', ascending=False).head(10)

            plt.figure(figsize=(12, 6))
            plt.bar(df_cov_sorted['node_name'], df_cov_sorted['Total_Duration_Min'], color='coral', edgecolor='black')
            plt.title('Top 10 Most Vulnerable Nodes (Avg Uncovered Min/Day - 700 Demand)', fontsize=14,
                      fontweight='bold')
            plt.xlabel('Node Name / Location', fontsize=12);
            plt.ylabel('Avg Uncovered Time per Day (Minutes)', fontsize=12)
            plt.xticks(rotation=45, ha='right');
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(SCRIPT_DIR / "chart_defense_vulnerable_nodes_cross_700.png", dpi=300, bbox_inches='tight');
            plt.close()

        df_failed = df_resp[df_resp['on_time'] == False].copy()
        if not df_failed.empty:
            df_failed['hour'] = df_failed['created_clock'].apply(lambda x: int(str(x).split(':')[0]))

            hourly_fails = df_failed['hour'].value_counts().sort_index() / num_seeds
            plt.figure(figsize=(10, 6))
            plt.bar(hourly_fails.index, hourly_fails.values, color='crimson', edgecolor='black')
            plt.title('Failed Missions by Hour of Day (Avg per Day - 700 Demand)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour of Day', fontsize=12);
            plt.ylabel('Avg Number of Failed Missions', fontsize=12)
            plt.xticks(range(0, 24));
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(SCRIPT_DIR / "chart_defense_fails_by_hour_cross_700.png", dpi=300, bbox_inches='tight');
            plt.close()

            df_failed['node_name'] = df_failed['node'].apply(get_node_name)
            node_fails = (df_failed['node_name'].value_counts() / num_seeds).head(10)
            plt.figure(figsize=(12, 6))
            plt.bar(node_fails.index, node_fails.values, color='darkred', edgecolor='black')
            plt.title('Top 10 Nodes with Highest Mission Failures (Avg per Day - 700 Demand)', fontsize=14,
                      fontweight='bold')
            plt.xlabel('Node Name / Location', fontsize=12);
            plt.ylabel('Avg Number of Failed Missions', fontsize=12)
            plt.xticks(rotation=45, ha='right');
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(SCRIPT_DIR / "chart_defense_fails_by_node_cross_700.png", dpi=300, bbox_inches='tight');
            plt.close()


if __name__ == "__main__":
    main()
