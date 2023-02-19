import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    n = len(y_true)
    y_true_pos = sum(y_true)
    y_pred_pos = sum(y_pred)
    tp = sum(y_pred[y_true])
    
    try:
        precision = tp / y_pred_pos
    except:
        precision = None

    try:
        recall = tp / y_true_pos
    except:
        recall = None

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = None
        
    try:
        accuracy = sum(y_pred == y_true) / n
    except:
        accuracy = None

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    n = len(y_true)
    mc_accuracy = sum(y_pred == y_true) / n
    return mc_accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    try:
        numerator = np.square(y_pred - y_true).sum()
        denominator = np.square(y_true - np.average(y_true)).sum()
        r2 = 1 - numerator / denominator
        return r2
    except:
        return None

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    
    try:
        n = len(y_true)
        mse = np.square(y_pred - y_true).sum() / n
        return mse
    except:
        return None


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    try:
        n = len(y_true)
        mae = np.abs(y_pred - y_true).sum() / n
        return mae
    except:
        return None