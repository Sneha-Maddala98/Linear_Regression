# Linear Regression Using Stochastic Gradient Descent

This Python script demonstrates linear regression using the stochastic gradient descent (SGD) optimization technique. Linear regression is a fundamental machine learning algorithm used for predicting a continuous target variable based on one or more input features. In this script, we will learn the coefficients of the linear model that best fits the data.

## Prerequisites

Before running the script, ensure that you have the following libraries installed:

- [NumPy](https://numpy.org/): For numerical operations.
- [Pandas](https://pandas.pydata.org/): For data manipulation.
- [argparse](https://docs.python.org/3/library/argparse.html): For command-line arguments.

You can install these libraries using pip:

```bash
pip install numpy pandas
```

## Usage

To run the script, use the following command:

```bash
python linear_regression_sgd.py -d data_file.csv -l learning_rate -t threshold
```

- `data_file.csv`: The path to the CSV file containing your dataset. The CSV file should have columns for data points, and each point should have an 'x' value as the feature and a 'y' value as the target variable.
- `learning_rate`: The learning rate for the stochastic gradient descent algorithm. It determines the step size during the optimization process.
- `threshold`: The threshold for convergence. The algorithm will stop when the change in the sum of squared errors (SSE) is smaller than this threshold.

## Algorithm

The script implements linear regression using stochastic gradient descent with the following steps:

1. Read the dataset from the specified CSV file.
2. Initialize the weights (`W`) with zeros.
3. Perform stochastic gradient descent to update the weights iteratively.
4. Calculate the predicted values (`f_x`) using the current weights.
5. Compute the gradient of the mean squared error (MSE) with respect to the weights.
6. Update the weights based on the gradient and the learning rate.
7. Repeat steps 4 to 6 until convergence, determined by the change in SSE.

## Example

Here is an example command to run the script:

```bash
python linear_regression_sgd.py -d data.csv -l 0.01 -t 0.001
```

This will load the dataset from `data.csv`, set the learning rate to 0.01, and stop the optimization when the change in SSE is less than 0.001.

The script will print the progress of the optimization, including the iteration, weights, and SSE. It will also display the final learned weights.

You can adjust the learning rate and threshold to fine-tune the optimization process for your specific dataset.

