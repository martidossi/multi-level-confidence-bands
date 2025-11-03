import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import norm


def calculate_coverage(y_true, lower_bound, upper_bound):
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage


def model_eval_metrics(y_true, y_pred):
    metrics = {
        'mse': mean_squared_error(y_true, y_pred), 
        'mape': round(mean_absolute_percentage_error(y_true, y_pred)*100, 2)
    }
    return metrics


# Function for 95% confidence interval using standard deviation of residuals
def get_confidence_interval(
        y_test_pred, 
        y_calib_pred, 
        y_calib, 
        alpha=0.05,
        ddof=1,
        bias_correct=False
        ):
    """
    Normal-based symmetric confidence intervals using calibration residuals.

    y_calib_pred, y_calib: calibration predictions and true values (hold-out/calibration set)
    y_test_pred: predictions to form intervals for
    alpha: miscoverage level (0.05 -> 95% CI)
    ddof: degrees of freedom for sample std (1 is recommended)
    bias_correct: if True, shift central prediction by mean residual (to correct bias)
    """
    residuals = y_calib - y_calib_pred # raw residuals (can be negative)

    # Compute the standard deviation of the residuals
    std_residuals = np.std(residuals, ddof=ddof)

    # Compute the margin of error using the z-score
    z_score = norm.ppf(1 - alpha/2) # for (1-alpha) conf. interval

    if bias_correct:
        mean_resid = np.mean(residuals)
        center = y_test_pred + mean_resid
    else:
        center = y_test_pred

    # Calculate lower and upper bounds
    lower_bound = center - z_score * std_residuals
    upper_bound = center + z_score * std_residuals

    return lower_bound, upper_bound


def get_conformalized_interval(y_test_pred, y_calib_pred, y_calib, alpha=0.05):
    # The method assumes symmetric intervals around the point prediction
    # What matters is only the magnitude of the error, not its direction (whether you under- or over-predicted)
    # abs residuals: because we want to measure how far off we are in either direction
    calib_residuals = np.abs(y_calib - y_calib_pred) 
    q_hat = np.quantile(calib_residuals, 1 - alpha)
    lower_bound = y_test_pred - q_hat
    upper_bound = y_test_pred + q_hat
    return lower_bound, upper_bound