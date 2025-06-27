Random Search Algorithms for Optimization
Overview
This project implements and evaluates random search algorithms for optimization problems, comparing their performance with gradient descent. The implementation includes both global random search and a modified random search with population-based sampling. These algorithms are tested on standard test functions and applied to hyperparameter tuning of a convolutional neural network (CNN) for the CIFAR-10 dataset.
Files

random_search_algorithms.py: Implementation of the optimization algorithms and test functions
main_experiment.py: Script to run experiments on the test functions
cnn_hyperparameter_tuning.py: Application of the algorithms to CNN hyperparameter tuning

Requirements

Python 3.6+
NumPy
TensorFlow/Keras
Matplotlib
scikit-learn

Installation
pip install numpy tensorflow matplotlib scikit-learn

Implementation Order
To implement the project, follow this sequence:

First, implement random_search_algorithms.py containing the core optimization algorithms and test functions
Next, implement main_experiment.py to test the algorithms on standard functions
Finally, implement cnn_hyperparameter_tuning.py to apply the algorithms to hyperparameter tuning

This order is important as each subsequent file depends on functions defined in the previous files.
Usage

Running the Test Function Experiments
python main_experiment.py
This will run the optimization algorithms on the two test functions and generate plots comparing their performance.

Running the CNN Hyperparameter Tuning
python cnn_hyperparameter_tuning.py
This will apply the random search algorithms to tune hyperparameters of a CNN model for the CIFAR-10 dataset.

Algorithms Implemented
Global Random Search
A simple approach that randomly samples points from the parameter space and keeps track of the best point found.

Modified Random Search
An enhanced algorithm that uses population-based sampling to focus exploration in promising regions of the parameter space.

Gradient Descent
Implemented for comparison purposes, using gradient information to navigate toward the minimum.

Test Functions
Function 1: f(x,y) = 6*(x-1)^4 + 8*(y-2)^2
Smooth, differentiable function with a global minimum at (1,2)

Function 2: f(x,y) = Max(x-1,0) + 8*|y-2|
Non-differentiable function with minima along the line where x ≤ 1 and y = 2

CNN Hyperparameter Tuning
The random search algorithms are applied to tune the following hyperparameters:

Batch size
Learning rate
Adam optimizer's β₁ and β₂ parameters
Number of training epochs

Output
The scripts generate various plots comparing the performance of the algorithms:

Cost function surface and contour plots
Cost vs. iterations plots
Cost vs. time plots
Search path visualizations
Hyperparameter tuning performance comparison