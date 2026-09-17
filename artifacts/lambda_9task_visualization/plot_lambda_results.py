import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


TASKS = [
    "Open\ndrawer",
    "Close\ndrawer",
    "Pick coke\ncan",
    "Move\nnear",
    "Place\napple",
    "Carrot on\nplate",
    "Eggplant in\nbasket",
    "Spoon on\ntowel",
    "Stack\ncube",
]

VANILLA = np.array([27, 56, 23, 53, 0, 5, 0, 0, 0], dtype=float)
LAMBDA_05 = np.array([50, 82, 44, 62, 1, 0, 3, 1, 0], dtype=float)
TASK_BEST = np.array([50, 82, 46, 68, 1, 9, 6, 5, 4], dtype=float)

LAMBDAS = np.array([0, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .75])
OVERALL_RATE = np.array([
    18.2, 21.4, 20.8, 22.2, 23.4, 22.9, 22.9,
    21.6, 24.9, 27.0, 25.3, 25.3, 23.1,
])


def label_bars(axis, bars, color, minimum=0):
    for bar in bars:
        value = bar.get_height()
        if value >= minimum:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.1,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )


plt.style.use("seaborn-whitegrid")
fig = plt.figure(figsize=(16, 10), dpi=180)
grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.30)

# Panel A: task-wise comparison.
axis_top = fig.add_subplot(grid[0])
x_positions = np.arange(len(TASKS))
bar_width = 0.25

bars_vanilla = axis_top.bar(
    x_positions - bar_width,
    VANILLA,
    bar_width,
    label="Vanilla (λ=0)",
    color="#9AA0A6",
    edgecolor="white",
    linewidth=0.8,
)
bars_l05 = axis_top.bar(
    x_positions,
    LAMBDA_05,
    bar_width,
    label="Fixed λ=0.5",
    color="#2563EB",
    edgecolor="white",
    linewidth=0.8,
)
bars_best = axis_top.bar(
    x_positions + bar_width,
    TASK_BEST,
    bar_width,
    label="Best λ per task",
    color="#F59E0B",
    edgecolor="white",
    linewidth=0.8,
)

label_bars(axis_top, bars_vanilla, "#5F6368")
label_bars(axis_top, bars_l05, "#174EA6")
label_bars(axis_top, bars_best, "#9A5B00")

axis_top.set_title(
    "Task-wise Success Rate: Vanilla vs Fixed λ=0.5 vs Per-task Best",
    fontsize=17,
    fontweight="bold",
    pad=16,
)
axis_top.set_ylabel("Success rate (%)", fontsize=12)
axis_top.set_xticks(x_positions)
axis_top.set_xticklabels(TASKS, fontsize=10)
axis_top.set_ylim(0, 92)
axis_top.legend(loc="upper right", frameon=True, ncol=3, fontsize=10)
axis_top.grid(axis="x", visible=False)
axis_top.text(
    0.01,
    0.96,
    "100 seeds per task",
    transform=axis_top.transAxes,
    fontsize=10,
    color="#5F6368",
    va="top",
)

# Panel B: overall fixed-lambda sweep and task-wise tuned upper line.
axis_bottom = fig.add_subplot(grid[1])
axis_bottom.plot(
    LAMBDAS,
    OVERALL_RATE,
    color="#64748B",
    linewidth=2.3,
    marker="o",
    markersize=5,
    label="One fixed λ for all tasks",
    zorder=2,
)

vanilla_index = int(np.where(LAMBDAS == 0)[0][0])
l05_index = int(np.where(LAMBDAS == .5)[0][0])
axis_bottom.scatter(
    [LAMBDAS[vanilla_index]],
    [OVERALL_RATE[vanilla_index]],
    s=150,
    color="#9AA0A6",
    edgecolor="white",
    linewidth=1.5,
    zorder=4,
    label="Vanilla: 18.2%",
)
axis_bottom.scatter(
    [LAMBDAS[l05_index]],
    [OVERALL_RATE[l05_index]],
    s=180,
    color="#2563EB",
    edgecolor="white",
    linewidth=1.5,
    zorder=5,
    label="Best fixed λ=0.5: 27.0%",
)
axis_bottom.axhline(
    30.1,
    color="#F59E0B",
    linewidth=2.5,
    linestyle="--",
    label="Best λ per task: 30.1%",
    zorder=1,
)

axis_bottom.annotate(
    "Vanilla\n164/900",
    xy=(0, 18.2),
    xytext=(0.035, 16.9),
    arrowprops=dict(arrowstyle="->", color="#5F6368", lw=1.2),
    fontsize=10,
    color="#5F6368",
    fontweight="bold",
)
axis_bottom.annotate(
    "Best fixed λ\n243/900",
    xy=(.5, 27.0),
    xytext=(.55, 28.1),
    arrowprops=dict(arrowstyle="->", color="#174EA6", lw=1.2),
    fontsize=10,
    color="#174EA6",
    fontweight="bold",
)
axis_bottom.text(
    .745,
    30.45,
    "Task-wise tuned: 271/900",
    ha="right",
    va="bottom",
    fontsize=10,
    color="#9A5B00",
    fontweight="bold",
)

axis_bottom.set_title(
    "Overall Success Rate Across λ (9 tasks × 100 seeds)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
axis_bottom.set_xlabel("Guidance strength λ", fontsize=12)
axis_bottom.set_ylabel("Overall success rate (%)", fontsize=12)
axis_bottom.set_xticks(LAMBDAS)
axis_bottom.set_xticklabels([f"{value:g}" for value in LAMBDAS], fontsize=9)
axis_bottom.set_xlim(-0.025, 0.775)
axis_bottom.set_ylim(15.5, 32.5)
axis_bottom.legend(loc="lower right", ncol=2, fontsize=9, frameon=True)
axis_bottom.grid(axis="x", visible=False)

fig.suptitle(
    "Prompt-CD λ Sweep on 9 Tasks",
    fontsize=21,
    fontweight="bold",
    y=0.985,
)
fig.text(
    0.5,
    0.012,
    "Per-task tuning improves 27.0% → 30.1% (+28 successes, +3.1 percentage points); values are selected on the same seed set.",
    ha="center",
    fontsize=10,
    color="#475569",
)

output_png = "/home/leju-suzhou/zjt_ws/token-cd/artifacts/lambda_9task_visualization/figure_lambda_comparison.png"
output_pdf = "/home/leju-suzhou/zjt_ws/token-cd/artifacts/lambda_9task_visualization/figure_lambda_comparison.pdf"
fig.savefig(output_png, bbox_inches="tight", facecolor="white")
fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
print(output_png)
print(output_pdf)
