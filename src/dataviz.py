import numpy as np


def true_vs_pred(y_true, y_pred, mse, ax):
    ax.scatter(y_true, y_pred, marker='o', s=10, color='royalblue', alpha=0.4)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--') 
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
    ax.set_title(f'Training set (mse={round(mse, 3)})')
    ax.grid()
    return ax

def plot_with_error_bars(
    y_true,
    y_pred,
    lower_bound,
    upper_bound,
    ax,
    xlabel='y true',
    ylabel='y pred',
    title=None,
    sample_frac=None,
    random_state=42
):
    """Plots predictions with error bars."""
    if sample_frac is not None:
        rng = np.random.default_rng(random_state)
        sampled_idx = rng.choice(len(y_pred), size=int(len(y_pred) * sample_frac), replace=False)
        y_true, y_pred = y_true[sampled_idx], y_pred[sampled_idx]
        lower_bound, upper_bound = lower_bound[sampled_idx], upper_bound[sampled_idx]

    lower_error = np.maximum(0, y_pred - lower_bound)
    upper_error = np.maximum(0, upper_bound - y_pred)
    ax.errorbar(
        y_true,
        y_pred,
        yerr=[lower_error, upper_error],
        fmt='o', markersize=5, color='blue', ecolor='red', alpha=0.5,
        label='Predictions with uncertainty interval'
    )
    # Matplotlib’s errorbar function expects yerr as the distance from the point to the ends of the error bar:
    # lower_error = distance from y_pred down to the lower bound → y_pred - lower_bound
    # upper_error = distance from y_pred up to the upper bound → upper_bound - y_pred
    ax.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        '--', color='black', 
        label='Ideal prediction'
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)