import numpy as np
import random
from typing import List, Tuple

# ANSI escape codes for highlighting
GREEN = '\033[92m'
ENDC = '\033[0m'
BLUE = '\033[94m'

# ==============================================================================
# I. LINEAR TREND UTILITY
# ==============================================================================

def calculate_linear_trend(marks: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the slope (m) and intercept (b) for a sequence of marks using
    Ordinary Least Squares (OLS), where X is the mark index (time).
    
    Args:
        marks (np.ndarray): 1D array of marks.
        
    Returns:
        Tuple[float, float]: (slope, intercept)
    """
    # Clip marks to ensure they are within the [0, 100] range for stable calculation
    marks = np.clip(marks, 0, 100)
    N = len(marks)
    
    # If less than 2 points, cannot compute a line, return 0 slope
    if N < 2:
        return 0.0, marks[0] if N == 1 else 50.0 # Default intercept to 50 if empty
    
    # Time indices (X) are 1, 2, 3, ..., N
    X = np.arange(1, N + 1).astype(np.float32)
    
    mean_x = np.mean(X)
    mean_y = np.mean(marks)
    
    # Analytical solution for slope (m)
    numerator = np.sum((X - mean_x) * (marks - mean_y))
    denominator = np.sum((X - mean_x)**2)
    
    m = numerator / denominator if denominator != 0 else 0.0
    
    # Analytical solution for intercept (b)
    b = mean_y - m * mean_x
    
    return m, b

# ==============================================================================
# II. SIMPLE PREDICTION FUNCTION
# ==============================================================================

def predict_best_fit(marks_list: List[float]) -> float:
    """
    Computes the line of best fit for the given marks and predicts the next mark
    (at time index N+1). The result is bounded between 0 and 100.
    
    Args:
        marks_list (List[float]): List of previous marks.
        
    Returns:
        float: The predicted next mark, bounded by [0, 100].
    """
    marks_array = np.array(marks_list, dtype=np.float32)
    N = len(marks_array)
    
    if N == 0:
        print(f"{BLUE}Warning: Empty mark sequence. Returning default 50.0.{ENDC}")
        return 50.0

    # 1. Compute Trend
    gradient, y_intercept = calculate_linear_trend(marks_array)
    
    # 2. Predict Next Mark
    # The next mark occurs at time index N + 1
    next_time_index = N + 1
    
    # Prediction formula: y = m*x + b
    predicted_mark = gradient * next_time_index + y_intercept
    
    # 3. Bound the Prediction
    # Ensure the mark is between 0 and 100
    bounded_prediction = np.clip(predicted_mark, 0.0, 100.0).item()
    
    print(f"\n--- Prediction Details ---")
    print(f"Input Marks [N={N}]: {marks_list}")
    print(f"Time Indices (X) : {list(range(1, N + 1))}")
    print(f"Gradient (m)     : {gradient:.2f}")
    print(f"Y-Intercept (b)  : {y_intercept:.2f}")
    print(f"Next X Index     : {next_time_index}")
    
    return bounded_prediction

# ==============================================================================
# III. EXAMPLE USAGE
# ==============================================================================

if __name__ == '__main__':
    
    # Example 1: Strongly Improving Trend (Expected > 90)
    marks_improving = [70, 80, 85, 90] 
    pred_improving = predict_best_fit(marks_improving)
    print(f"Predicted Next Mark (Improving): {GREEN}{pred_improving:.2f}{ENDC}")
    print("-" * 50)

    # Example 2: Steadily Declining Trend (Expected < 40)
    marks_declining = [90, 75, 60, 45]
    pred_declining = predict_best_fit(marks_declining)
    print(f"Predicted Next Mark (Declining): {GREEN}{pred_declining:.2f}{ENDC}")
    print("-" * 50)
    
    # Example 3: Consistent Marks (Expected near 60)
    marks_consistent = [60, 62, 58, 60, 61]
    pred_consistent = predict_best_fit(marks_consistent)
    print(f"Predicted Next Mark (Consistent): {GREEN}{pred_consistent:.2f}{ENDC}")
    print("-" * 50)

    # Example 4: Edge Case - Prediction beyond 100 (Should be capped)
    marks_soaring = [90, 95, 98, 99]
    pred_soaring = predict_best_fit(marks_soaring)
    print(f"Predicted Next Mark (Soaring/Capped): {GREEN}{pred_soaring:.2f}{ENDC} (Capped at 100.00)")
    print("-" * 50)

    # Example 5: Edge Case - Prediction below 0 (Should be capped)
    marks_plummeting = [50, 30, 10]
    pred_plummeting = predict_best_fit(marks_plummeting)
    print(f"Predicted Next Mark (Plummeting/Capped): {GREEN}{pred_plummeting:.2f}{ENDC} (Capped at 0.00)")
    print("-" * 50)