import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Import functions from random_search_algorithms.py
from random_search_algorithms import (
    global_random_search, modified_random_search, gradient_descent,
    function1, function1_gradient, function2, function2_gradient,
    plot_cost_landscape, plot_search_paths, compare_methods
)

def run_test_functions_experiments():
    """
    Run experiments on test functions and compare optimization methods.
    """
    print("Running experiments on test functions...")
    
    # Define parameter bounds for test functions
    # Function 1: f(x,y) = 6*(x-1)^4 + 8*(y-2)^2 has minimum at (1,2)
    function1_bounds = [(-1, 3), (0, 4)]
    
    # Function 2: f(x,y) = Max(x-1,0) + 8*|y-2| has minimum at (x≤1, y=2)
    function2_bounds = [(-1, 3), (0, 4)]
    
    # Plot cost landscapes
    print("Plotting cost landscapes...")
    fig_func1, X_func1, Y_func1, Z_func1 = plot_cost_landscape(function1, function1_bounds)
    plt.savefig('function1_landscape.png')
    plt.close(fig_func1)
    
    fig_func2, X_func2, Y_func2, Z_func2 = plot_cost_landscape(function2, function2_bounds)
    plt.savefig('function2_landscape.png')
    plt.close(fig_func2)
    
    # Compare methods on Function 1
    print("\nComparing methods on Function 1: f(x,y) = 6*(x-1)^4 + 8*(y-2)^2")
    function1_results = compare_methods(function1, function1_gradient, function1_bounds, 2, n_runs=5)
    
    # Plot cost vs iterations
    plt.figure(figsize=(10, 6))
    for method_name, result in function1_results.items():
        plt.plot(range(len(result['avg_costs'])), result['avg_costs'], label=method_name)
        plt.fill_between(
            range(len(result['avg_costs'])),
            result['avg_costs'] - result['std_costs'],
            result['avg_costs'] + result['std_costs'],
            alpha=0.2
        )
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations - Function 1: f(x,y) = 6*(x-1)^4 + 8*(y-2)^2')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('function1_cost_vs_iterations.png')
    plt.close()
    
    # Plot cost vs time
    plt.figure(figsize=(10, 6))
    for method_name, result in function1_results.items():
        plt.plot(result['avg_times'], result['avg_costs'], label=method_name)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cost')
    plt.title('Cost vs Time - Function 1: f(x,y) = 6*(x-1)^4 + 8*(y-2)^2')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('function1_cost_vs_time.png')
    plt.close()
    
    # Plot search paths
    fig_func1_paths = plot_search_paths(function1, function1_bounds, function1_results)
    plt.savefig('function1_search_paths.png')
    plt.close(fig_func1_paths)
    
    # Compare methods on Function 2
    print("\nComparing methods on Function 2: f(x,y) = Max(x-1,0) + 8*|y-2|")
    function2_results = compare_methods(function2, function2_gradient, function2_bounds, 2, n_runs=5)
    
    # Plot cost vs iterations
    plt.figure(figsize=(10, 6))
    for method_name, result in function2_results.items():
        plt.plot(range(len(result['avg_costs'])), result['avg_costs'], label=method_name)
        plt.fill_between(
            range(len(result['avg_costs'])),
            result['avg_costs'] - result['std_costs'],
            result['avg_costs'] + result['std_costs'],
            alpha=0.2
        )
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations - Function 2: f(x,y) = Max(x-1,0) + 8*|y-2|')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('function2_cost_vs_iterations.png')
    plt.close()
    
    # Plot cost vs time
    plt.figure(figsize=(10, 6))
    for method_name, result in function2_results.items():
        plt.plot(result['avg_times'], result['avg_costs'], label=method_name)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cost')
    plt.title('Cost vs Time - Function 2: f(x,y) = Max(x-1,0) + 8*|y-2|')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('function2_cost_vs_time.png')
    plt.close()
    
    # Plot search paths
    fig_func2_paths = plot_search_paths(function2, function2_bounds, function2_results)
    plt.savefig('function2_search_paths.png')
    plt.close(fig_func2_paths)
    
    return function1_results, function2_results

if __name__ == "__main__":
    run_test_functions_experiments()