import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Calculate performance metrics for binary classification."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    # Calculate additional metrics
    metrics["false_positive_rate"] = fp / (fp + tn)
    metrics["false_negative_rate"] = fn / (fn + tp)
    metrics["true_positive_rate"] = tp / (tp + fn)
    metrics["true_negative_rate"] = tn / (tn + fp)

    return metrics


def calculate_inference_metrics(times: list, memory_usage: list) -> dict:
    """Calculate inference performance metrics."""
    return {
        "inference_time": {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
        },
        "memory_usage": {
            "mean": np.mean(memory_usage),
            "std": np.std(memory_usage),
            "min": np.min(memory_usage),
            "max": np.max(memory_usage),
            "p95": np.percentile(memory_usage, 95),
            "p99": np.percentile(memory_usage, 99),
        },
    }


def calculate_throughput(times: list, batch_size: int) -> float:
    """Calculate throughput in samples per second."""
    return batch_size / np.mean(times)


def calculate_latency(times: list) -> dict:
    """Calculate latency metrics in milliseconds."""
    return {
        "mean": np.mean(times) * 1000,
        "std": np.std(times) * 1000,
        "min": np.min(times) * 1000,
        "max": np.max(times) * 1000,
        "p95": np.percentile(times, 95) * 1000,
        "p99": np.percentile(times, 99) * 1000,
    } 