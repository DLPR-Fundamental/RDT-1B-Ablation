import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re, os

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the success rate data file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save plots and summary")
    return parser.parse_args()

def load_txt_data(input_file):
    rows = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.findall(r"(\w+)=([\w\.\-]+)", line)
            if match:
                row = dict(match)
                row["seed"] = int(row["seed"])
                row["num"] = int(row["num"])
                row["success_rate"] = float(row["success_rate"])
                rows.append(row)
    return pd.DataFrame(rows)

def visualize_all(df, output_path):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x="env", y="success_rate", data=df, errorbar='sd', palette="viridis", ax=ax1)
    ax1.set_title("Bar Plot (mean ± std)")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x="env", y="success_rate", data=df, palette="Set2", ax=ax2)
    ax2.set_title("Box Plot (Distribution)")
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    task_mean = df.groupby("env")["success_rate"].mean()
    tasks = task_mean.index.tolist()
    values = task_mean.values.tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]
    ax3.plot(angles, values, color='teal', linewidth=2)
    ax3.fill(angles, values, color='teal', alpha=0.25)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(tasks, fontsize=9)
    ax3.set_yticks([20, 40, 60, 80, 100])
    ax3.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax3.set_title("Radar Chart (Mean Success Rate)", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_summary(df, output_path):
    summary = df.groupby("env")["success_rate"].agg(["mean", "std", "min", "max"])
    summary.to_csv(output_path)
    print(f"Summary saved to: {output_path}")
    return summary

if __name__ == "__main__":
    args = parse_arg()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_txt_data(args.input_file)
    summary_path = os.path.join(args.output_dir, "success_rate_summary.csv")
    plot_path = os.path.join(args.output_dir, "success_rate_plots.png")
    
    summary = save_summary(df, summary_path)
    visualize_all(df, plot_path)
    
    print(f"Plot saved to: {plot_path}")