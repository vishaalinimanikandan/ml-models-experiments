import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Global Random Search
def global_random_search(cost_function, n_params, param_bounds, n_samples, verbose=False):
    """
    Implements global random search algorithm.
    
    Parameters:
    cost_function (callable): The function to minimize
    n_params (int): Number of parameters
    param_bounds (list): List of tuples (min, max) for each parameter
    n_samples (int): Number of samples to evaluate
    verbose (bool): Whether to print progress
    
    Returns:
    best_params (numpy.ndarray): Best parameters found
    best_cost (float): Best cost function value
    costs (list): Cost function values at each iteration
    times (list): Cumulative time at each iteration
    n_evals (int): Number of function evaluations
    """
    best_params = None
    best_cost = float('inf')
    costs = []
    times = []
    start_time = time.time()
    
    for i in range(n_samples):
        # Generate random parameters
        params = np.zeros(n_params)
        for j in range(n_params):
            lower, upper = param_bounds[j]
            params[j] = np.random.uniform(lower, upper)
        
        # Evaluate cost function
        cost = cost_function(params)
        
        # Update best parameters if cost is lower
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        
        # Record cost and time
        costs.append(best_cost)
        times.append(time.time() - start_time)
        
        if verbose and (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{n_samples}, Best cost: {best_cost:.6f}")
    
    return best_params, best_cost, costs, times, n_samples

# Modified Random Search with Population-Based Sampling
def modified_random_search(cost_function, n_params, param_bounds, n_samples, m_best, n_iterations, neighborhood_size=0.1, verbose=False):
    """
    Implements modified random search with population-based sampling.
    
    Parameters:
    cost_function (callable): The function to minimize
    n_params (int): Number of parameters
    param_bounds (list): List of tuples (min, max) for each parameter
    n_samples (int): Number of samples per iteration
    m_best (int): Number of best points to keep
    n_iterations (int): Number of iterations
    neighborhood_size (float): Size of neighborhood for sampling
    verbose (bool): Whether to print progress
    
    Returns:
    best_params (numpy.ndarray): Best parameters found
    best_cost (float): Best cost function value
    costs (list): Cost function values at each iteration
    times (list): Cumulative time at each iteration
    n_evals (int): Number of function evaluations
    """
    # Initialize population with random points
    population = []
    population_costs = []
    costs = []
    times = []
    start_time = time.time()
    function_evals = 0
    
    # Initial random sampling
    for _ in range(n_samples):
        params = np.zeros(n_params)
        for j in range(n_params):
            lower, upper = param_bounds[j]
            params[j] = np.random.uniform(lower, upper)
        
        cost = cost_function(params)
        function_evals += 1
        
        population.append(params)
        population_costs.append(cost)
    
    # Sort population by cost and keep M best
    sorted_indices = np.argsort(population_costs)
    population = [population[i] for i in sorted_indices[:m_best]]
    population_costs = [population_costs[i] for i in sorted_indices[:m_best]]
    
    best_params = population[0].copy()
    best_cost = population_costs[0]
    
    costs.append(best_cost)
    times.append(time.time() - start_time)
    
    # Main iteration loop
    for iteration in range(n_iterations):
        new_population = []
        new_population_costs = []
        
        # Generate neighborhood samples around each point in the population
        for i, params in enumerate(population):
            for _ in range(n_samples // m_best):
                # Generate random perturbation
                new_params = params.copy()
                for j in range(n_params):
                    lower, upper = param_bounds[j]
                    range_size = upper - lower
                    perturbation = np.random.uniform(-neighborhood_size * range_size, 
                                                     neighborhood_size * range_size)
                    new_params[j] += perturbation
                    
                    # Clip to bounds
                    new_params[j] = max(lower, min(upper, new_params[j]))
                
                # Evaluate cost function
                cost = cost_function(new_params)
                function_evals += 1
                
                new_population.append(new_params)
                new_population_costs.append(cost)
        
        # Add current population to new samples
        new_population.extend(population)
        new_population_costs.extend(population_costs)
        
        # Sort combined population by cost and keep M best
        sorted_indices = np.argsort(new_population_costs)
        population = [new_population[i] for i in sorted_indices[:m_best]]
        population_costs = [new_population_costs[i] for i in sorted_indices[:m_best]]
        
        # Update best parameters if needed
        if population_costs[0] < best_cost:
            best_cost = population_costs[0]
            best_params = population[0].copy()
        
        costs.append(best_cost)
        times.append(time.time() - start_time)
        
        if verbose and (iteration+1) % 5 == 0:
            print(f"Iteration {iteration+1}/{n_iterations}, Best cost: {best_cost:.6f}")
    
    return best_params, best_cost, costs, times, function_evals

# Gradient Descent
def gradient_descent(cost_function, gradient_function, initial_params, learning_rate, n_iterations, verbose=False):
    """
    Implements gradient descent algorithm.
    
    Parameters:
    cost_function (callable): The function to minimize
    gradient_function (callable): The gradient of the cost function
    initial_params (numpy.ndarray): Initial parameters
    learning_rate (float): Learning rate
    n_iterations (int): Number of iterations
    verbose (bool): Whether to print progress
    
    Returns:
    best_params (numpy.ndarray): Best parameters found
    best_cost (float): Best cost function value
    costs (list): Cost function values at each iteration
    times (list): Cumulative time at each iteration
    n_evals (int): Number of function and gradient evaluations
    """
    params = initial_params.copy()
    costs = []
    times = []
    start_time = time.time()
    function_evals = 0
    gradient_evals = 0
    
    for i in range(n_iterations):
        # Evaluate cost function
        cost = cost_function(params)
        function_evals += 1
        
        # Record cost and time
        costs.append(cost)
        times.append(time.time() - start_time)
        
        # Compute gradient
        grad = gradient_function(params)
        gradient_evals += 1
        
        # Update parameters
        params = params - learning_rate * grad
        
        if verbose and (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Cost: {cost:.6f}")
    
    # Final evaluation
    final_cost = cost_function(params)
    function_evals += 1
    
    return params, final_cost, costs, times, (function_evals, gradient_evals)

# Define the two functions from Week 4
def function1(x):
    """Function 1: f(x,y) = 6*(x-1)^4 + 8*(y-2)^2"""
    return 6 * (x[0]-1)**4 + 8 * (x[1]-2)**2

def function1_gradient(x):
    """Gradient of function1"""
    dx = 24 * (x[0]-1)**3
    dy = 16 * (x[1]-2)
    return np.array([dx, dy])

def function2(x):
    """Function 2: f(x,y) = Max(x-1,0) + 8*|y-2|"""
    return max(x[0]-1, 0) + 8 * abs(x[1]-2)

def function2_gradient(x):
    """Gradient of function2 (approximation for non-differentiable points)"""
    dx = 1 if x[0] > 1 else 0
    dy = 8 if x[1] > 2 else -8 if x[1] < 2 else 0
    return np.array([dx, dy])

# Helper function to visualize the cost landscape
def plot_cost_landscape(cost_function, bounds, resolution=100):
    """
    Plots the cost landscape of a 2D function.
    
    Parameters:
    cost_function (callable): The function to plot
    bounds (list): List of tuples (min, max) for each parameter
    resolution (int): Number of points along each dimension
    
    Returns:
    fig (matplotlib.figure.Figure): The figure object
    X, Y (numpy.ndarray): Coordinate meshgrid
    Z (numpy.ndarray): Function values
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    X, Y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                       np.linspace(y_min, y_max, resolution))
    Z = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = cost_function(np.array([X[i, j], Y[i, j]]))
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax1.set_title('Cost Function Surface')
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, 20, cmap=cm.viridis)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Cost Function Contour')
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    
    return fig, X, Y, Z

# Function to visualize the search paths
def plot_search_paths(cost_function, bounds, methods_results, resolution=100):
    """
    Plots the search paths of different optimization methods.
    
    Parameters:
    cost_function (callable): The function being optimized
    bounds (list): List of tuples (min, max) for each parameter
    methods_results (dict): Dictionary of results for each method
    resolution (int): Number of points along each dimension
    
    Returns:
    fig (matplotlib.figure.Figure): The figure object
    """
    fig, X, Y, Z = plot_cost_landscape(cost_function, bounds, resolution)
    ax = fig.axes[1]  # Get the contour plot axis
    
    # Plot the search paths
    for method_name, results in methods_results.items():
        if method_name == 'Gradient Descent':
            # For gradient descent, we have the full path
            path = results['path']
            ax.plot(path[:, 0], path[:, 1], 'o-', label=method_name, markersize=4)
        else:
            # For random search methods, we plot the best points found
            path = results['path']
            ax.scatter(path[:, 0], path[:, 1], label=method_name, s=10)
    
    ax.legend()
    plt.tight_layout()
    
    return fig

# Function to compare optimization methods
def compare_methods(cost_function, gradient_function, bounds, n_params, n_runs=10):
    """
    Compares different optimization methods on a given cost function.
    
    Parameters:
    cost_function (callable): The function to minimize
    gradient_function (callable): The gradient of the cost function
    bounds (list): List of tuples (min, max) for each parameter
    n_params (int): Number of parameters
    n_runs (int): Number of runs to average over
    
    Returns:
    results (dict): Dictionary of results for each method
    """
    methods = {
        'Global Random Search': {
            'params': {
                'n_samples': 1000,
                'verbose': False
            },
            'costs': [],
            'times': [],
            'evals': []
        },
        'Modified Random Search': {
            'params': {
                'n_samples': 100,
                'm_best': 5,
                'n_iterations': 10,
                'neighborhood_size': 0.1,
                'verbose': False
            },
            'costs': [],
            'times': [],
            'evals': []
        },
        'Gradient Descent': {
            'params': {
                'learning_rate': 0.001,
                'n_iterations': 1000,
                'verbose': False
            },
            'costs': [],
            'times': [],
            'evals': []
        }
    }
    
    for method_name, method_info in methods.items():
        print(f"Running {method_name}...")
        
        for run in range(n_runs):
            if method_name == 'Global Random Search':
                best_params, best_cost, costs, times, n_evals = global_random_search(
                    cost_function,
                    n_params,
                    bounds,
                    **method_info['params']
                )
            elif method_name == 'Modified Random Search':
                best_params, best_cost, costs, times, n_evals = modified_random_search(
                    cost_function,
                    n_params,
                    bounds,
                    **method_info['params']
                )
            elif method_name == 'Gradient Descent':
                # For gradient descent, we use an initial point in the middle of the bounds
                initial_params = np.zeros(n_params)
                for j in range(n_params):
                    lower, upper = bounds[j]
                    initial_params[j] = (lower + upper) / 2
                
                best_params, best_cost, costs, times, n_evals = gradient_descent(
                    cost_function,
                    gradient_function,
                    initial_params,
                    **method_info['params']
                )
            
            # Store results
            method_info['costs'].append(costs)
            method_info['times'].append(times)
            method_info['evals'].append(n_evals)
            
            print(f"  Run {run+1}/{n_runs}, Best cost: {best_cost:.6f}")
    
    # Process results
    results = {}
    
    for method_name, method_info in methods.items():
        costs_array = np.array(method_info['costs'])
        times_array = np.array(method_info['times'])
        
        # Compute average and standard deviation
        avg_costs = np.mean(costs_array, axis=0)
        std_costs = np.std(costs_array, axis=0)
        avg_times = np.mean(times_array, axis=0)
        
        # For gradient descent, we need to compute the parameter path
        if method_name == 'Gradient Descent':
            # Use the last run for visualization
            path = np.zeros((method_info['params']['n_iterations'] + 1, n_params))
            initial_params = np.zeros(n_params)
            for j in range(n_params):
                lower, upper = bounds[j]
                initial_params[j] = (lower + upper) / 2
            
            path[0] = initial_params
            params = initial_params.copy()
            
            for i in range(method_info['params']['n_iterations']):
                grad = gradient_function(params)
                params = params - method_info['params']['learning_rate'] * grad
                path[i+1] = params
        else:
            # For random search methods, we use the best points found
            if method_name == 'Global Random Search':
                path = np.zeros((method_info['params']['n_samples'], n_params))
            else:
                path = np.zeros((method_info['params']['n_iterations'] + 1, n_params))
            
            # Generate random path for visualization
            best_cost = float('inf')
            if method_name == 'Global Random Search':
                for i in range(method_info['params']['n_samples']):
                    params = np.zeros(n_params)
                    for j in range(n_params):
                        lower, upper = bounds[j]
                        params[j] = np.random.uniform(lower, upper)
                    
                    cost = cost_function(params)
                    
                    if cost < best_cost:
                        best_cost = cost
                        path[i] = params
            else:
                # Initialize population with random points
                population = []
                population_costs = []
                
                for _ in range(method_info['params']['n_samples']):
                    params = np.zeros(n_params)
                    for j in range(n_params):
                        lower, upper = bounds[j]
                        params[j] = np.random.uniform(lower, upper)
                    
                    cost = cost_function(params)
                    
                    population.append(params)
                    population_costs.append(cost)
                
                # Sort population by cost and keep M best
                sorted_indices = np.argsort(population_costs)
                population = [population[i] for i in sorted_indices[:method_info['params']['m_best']]]
                population_costs = [population_costs[i] for i in sorted_indices[:method_info['params']['m_best']]]
                
                path[0] = population[0]
                
                for iteration in range(method_info['params']['n_iterations']):
                    new_population = []
                    new_population_costs = []
                    
                    for params in population:
                        for _ in range(method_info['params']['n_samples'] // method_info['params']['m_best']):
                            new_params = params.copy()
                            for j in range(n_params):
                                lower, upper = bounds[j]
                                range_size = upper - lower
                                perturbation = np.random.uniform(-method_info['params']['neighborhood_size'] * range_size, 
                                                              method_info['params']['neighborhood_size'] * range_size)
                                new_params[j] += perturbation
                                new_params[j] = max(lower, min(upper, new_params[j]))
                            
                            cost = cost_function(new_params)
                            
                            new_population.append(new_params)
                            new_population_costs.append(cost)
                    
                    new_population.extend(population)
                    new_population_costs.extend(population_costs)
                    
                    sorted_indices = np.argsort(new_population_costs)
                    population = [new_population[i] for i in sorted_indices[:method_info['params']['m_best']]]
                    population_costs = [new_population_costs[i] for i in sorted_indices[:method_info['params']['m_best']]]
                    
                    path[iteration+1] = population[0]
        
        results[method_name] = {
            'avg_costs': avg_costs,
            'std_costs': std_costs,
            'avg_times': avg_times,
            'path': path
        }
    
    return results