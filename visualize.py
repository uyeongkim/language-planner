import matplotlib.pyplot as plt
import numpy as np

def viz_correct(ax, k, pred_score, pred_error, baseline):
    """visualize correctness score plotting

    Args:
        ax (plt.axes): subplot to draw
        k (list): # of examples
        pred_score (list): correctness score
        pred_error (list): ratio of not executable plans
        baseline (float): roberta baseline correctness score
    """
    
    ax.bar(k, pred_score, width=1, label='correct')
    ax.bar(k, 100-np.array(pred_error)-np.array(pred_score), width=1, bottom=pred_score, label='executable', fill=False, hatch='xx')
    ax.hlines(baseline, xmin=1, xmax=20, linestyles='dotted', colors='black', label='baseline')
    ax.set_xlabel('# examples')
    ax.legend(loc='lower right')
    
    return

x = [4, 10, 20]
seen_y = [64.19, 78.19, 80.67]
unseen_y = [66.27, 78.39, 82.48]
seen_error = [17.10, 5.45, 1.98]
unseen_error = [18.58, 5.27, 1.84]

unseen_baseline = 73.78
seen_baseline = 75.09

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 4))
fig.suptitle('Correctness')
plt.xticks(x)
plt.ylim((55, 100))

ax0.set_title('Valid Seen')
viz_correct(ax0, x, seen_y, seen_error, seen_baseline)

ax1.set_title('Valid Unseen')
viz_correct(ax1, x, unseen_y, unseen_error, unseen_baseline)

plt.savefig('images/correctness.png')