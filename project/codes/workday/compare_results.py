import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import stats

SCRIPT_DIR = Path(os.getcwd())
FILE_BASELINE = SCRIPT_DIR / "response_summary_2.csv"
FILE_RL = SCRIPT_DIR / "response_summary_rl.csv"


def categorize_traffic(clock_str):
    try:
        hour = int(str(clock_str).split(':')[0])
        if (6 <= hour < 9) or (16 <= hour < 20):
            return 'Peak Traffic (Heavy)'
        else:
            return 'Normal/Light Traffic'
    except:
        return 'Normal/Light Traffic'


def main():
    if not FILE_BASELINE.exists() or not FILE_RL.exists():
        print("Error: Files not found.")
        return

    df_base = pd.read_csv(FILE_BASELINE)
    df_rl = pd.read_csv(FILE_RL)

    df_base['Traffic_Condition'] = df_base['created_clock'].apply(categorize_traffic)
    df_rl['Traffic_Condition'] = df_rl['created_clock'].apply(categorize_traffic)

    base_seed_stats = df_base.groupby(['seed', 'Traffic_Condition'])['on_time'].mean().unstack() * 100
    rl_seed_stats = df_rl.groupby(['seed', 'Traffic_Condition'])['on_time'].mean().unstack() * 100

    conditions = ['Normal/Light Traffic', 'Peak Traffic (Heavy)']
    base_means, base_stds = [], []
    rl_means, rl_stds = [], []
    p_values = []

    for cond in conditions:
        b_data = base_seed_stats[cond].dropna()
        r_data = rl_seed_stats[cond].dropna()

        base_means.append(b_data.mean());
        base_stds.append(b_data.std())
        rl_means.append(r_data.mean());
        rl_stds.append(r_data.std())

        t_stat, p_val = stats.ttest_rel(r_data, b_data)
        p_values.append(p_val)

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width / 2, base_means, width, yerr=base_stds, capsize=5, label='Dynamic Baseline (No AI)',
                    color='#e74c3c', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width / 2, rl_means, width, yerr=rl_stds, capsize=5, label='Trained Intelligent Agent',
                    color='#2ecc71', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Success Rate (On-Time %)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of RL in Different Traffic Conditions (30-Run Statistical Avg)', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x);
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylim(0, 115);
    ax.legend(fontsize=12, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, rect in enumerate(rects1):
        ax.annotate(f'{rect.get_height():.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + base_stds[i]), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i, rect in enumerate(rects2):
        ax.annotate(f'{rect.get_height():.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + rl_stds[i]), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

    for i in range(len(conditions)):
        p_text = f"p < 0.001" if p_values[i] < 0.001 else f"p = {p_values[i]:.3f}"
        max_h = max(base_means[i] + base_stds[i], rl_means[i] + rl_stds[i])
        ax.plot([x[i] - width / 2, x[i] + width / 2], [max_h + 5, max_h + 5], color='black', lw=1.5)
        ax.plot([x[i] - width / 2, x[i] - width / 2], [max_h + 3, max_h + 5], color='black', lw=1.5)
        ax.plot([x[i] + width / 2, x[i] + width / 2], [max_h + 3, max_h + 5], color='black', lw=1.5)
        ax.text(x[i], max_h + 6, p_text, ha='center', va='bottom', fontsize=11, color='blue', fontweight='bold')

    save_path = SCRIPT_DIR / "chart_comparison_baseline_vs_rl_STATISTICAL.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {save_path.name}")
    plt.close()

    table_data = {
        "Traffic Condition": conditions,
        "Baseline (Mean ± Std)": [f"{base_means[0]:.2f}% ± {base_stds[0]:.2f}%",
                                  f"{base_means[1]:.2f}% ± {base_stds[1]:.2f}%"],
        "RL Agent (Mean ± Std)": [f"{rl_means[0]:.2f}% ± {rl_stds[0]:.2f}%", f"{rl_means[1]:.2f}% ± {rl_stds[1]:.2f}%"],
        "P-value": [f"{p_values[0]:.4e}", f"{p_values[1]:.4e}"]
    }
    df_table = pd.DataFrame(table_data)
    df_table.to_csv(SCRIPT_DIR / "statistical_comparison_table.csv", index=False)
    print("\n--- Formal Academic Table ---")
    print(df_table.to_markdown(index=False))


if __name__ == "__main__":
    main()
