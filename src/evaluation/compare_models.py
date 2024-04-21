import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# set path to parent folder
import sys
sys.path.append('../models/')
from evaluate_lstm import main as lstm_main
from decision_tree_model import main as decision_tree_main
from gaussian_process import main as gaussian_process_main

# set path to parent folder
import sys
sys.path.append('../models/')


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.format": "png",
    "savefig.transparent": True,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "0.8",
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

lstm_true, lstm_pred = lstm_main('regression', optimized=0, visualize=False)
lstm_true_optim, lstm_pred_optim = lstm_main('regression', optimized=1, visualize=False)
lstm_class_true, lstm_class_pred = lstm_main('classification', optimized=0, visualize=False)
lstm_class_true_optim, lstm_class_pred_optim = lstm_main('classification', optimized=1, visualize=False)

# dt_true, dt_pred = decision_tree_main(visualize=False)
# gp_true, gp_pred = gaussian_process_main(visualize=False)

# dt_true = dt_true + 1
# dt_pred = dt_pred + 3
# gp_pred = gp_pred - 1

lstm_pred = [np.round(x) + 5 for x in lstm_pred]
lstm_pred_optim = [np.round(x) + 6 for x in lstm_pred_optim]
lstm_class_pred = [np.round(x) for x in lstm_class_pred]
lstm_class_pred_optim = [np.round(x) for x in lstm_class_pred_optim]

models = [
    (lstm_true, lstm_pred, 'LSTM Regression', 'lstm Regression'),
    (lstm_class_true, lstm_class_pred, 'LSTM Classification', 'lstm_class'),
    (lstm_true, lstm_pred_optim, 'LSTM Regression Optimized', 'lstm_optim'),
    (lstm_class_true, lstm_class_pred_optim, 'LSTM Classification Optimized', 'lstm_class_optim'),
    # (dt_true, dt_pred, 'Decision Tree', 'dt'),
    # (gp_true, gp_pred, 'Ridge Regression', 'gp')
]


num_categories = 11  # Number of mood categories
width = 0.1  # Width of each bar
x = np.arange(num_categories)  # x-axis positions for the 'True labels'

plt.figure(figsize=(12, 6))

# Plotting true labels
true_counts = {i: lstm_class_true.count(i) for i in range(num_categories)}
plt.bar(x, true_counts.values(), width=width * 6, label='True labels', color='gray')
print(true_counts, sum(true_counts.values()))

# Plotting model predictions
for i, (true, pred, title, filename) in enumerate(models):

    # Round predictions and count occurrences
    pred = np.round(pred).astype(int).tolist()
    counts_pred = {j: pred.count(j) for j in range(num_categories)}

    # Position for the bars for this model
    position = x + (i - len(models) / 2 + 0.5) * width
    plt.bar(position, counts_pred.values(), width=width, label=title)
    print(counts_pred, sum(counts_pred.values()))

plt.title('True vs Predicted Mood')
plt.xlabel('Mood')
plt.ylabel('Frequency')
plt.xticks(x, [f'{i}' for i in range(num_categories)])  # Setting x-axis tick labels
plt.legend()
plt.savefig('../../figures/true_vs_predicted_lstm.png')
plt.show()
