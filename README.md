# Gradient-Boosting-Decision-Tree-Regressor

This project demonstrates the implementation of Gradient Boosting using Decision Trees to predict a quadratic function. The code fits residuals iteratively, plotting the results after each boosting step.

## Features:
- Iterative gradient boosting approach.
- Uses `DecisionTreeRegressor` from `scikit-learn` to fit residuals at each stage.
- Visualizes the prediction improvement across iterations.

## Requirements:
- Python 3.x
- `numpy`
- `matplotlib`
- `scikit-learn`

## How to Run:
1. Install the required dependencies:
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
2. Run the Python script:
    ```bash
    python gradient_boosting.py
    ```

## How It Works:
1. The code generates 100 random data points with noise based on a quadratic function.
2. At each iteration, a `DecisionTreeRegressor` is fitted to the residual errors from the previous prediction.
3. The process is repeated for 5 iterations, and the improvement in the predictions is shown via plots after each step.

## Example Output:
The following is an example of the output after the gradient boosting iterations:

![Example Plot](example_plot.png)
