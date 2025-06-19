
## Heaviside function theta
import sympy as sp

# Define symbols
x, y = sp.symbols('x y')
# Define the first function
f1 = 6 * (x - 1)**4 + 8 * (y - 2)**2

# Compute the partial derivative with respect to x
df1_dx = sp.diff(f1, x)
print("Partial derivative of f1 with respect to x:", df1_dx)
# Compute the partial derivative with respect to y
df1_dy = sp.diff(f1, y)
print("Partial derivative of f1 with respect to y:", df1_dy)


# Define the second function
f2 = sp.Max(x - 1, 0) + 8 * sp.Abs(y - 2)

# Compute the partial derivative with respect to x
df2_dx = sp.diff(f2, x)
print("Partial derivative of f2 with respect to x:", df2_dx)
# Compute the partial derivative with respect to y
df2_dy = sp.diff(f2, y)
print("Partial derivative of f2 with respect to y:", df2_dy)


# ## part a


## Polyak step size algorithm 
import numpy as np

def polyak_step_size(gradient, step_size):
    """
    Compute the update step using the Polyak step size.
    """
    return -step_size * gradient

def gradient_descent_polyak(f, grad_f, x0, step_size, max_iterations=1000, tol=1e-6):
    """
    Perform gradient descent using the Polyak step size.

    Parameters:
    - f: The function to minimize.
    - grad_f: The gradient of the function.
    - x0: Initial point (numpy array).
    - step_size: Step size (learning rate).
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.

    Returns:
    - x: The final point after optimization.
    - history: List of function values at each iteration.
    """
    x = x0  # Initialize the current point
    history = []  # Store the function values at each iteration

    for iteration in range(max_iterations):
        gradient = grad_f(x)  # Compute the gradient at the current point
        update = polyak_step_size(gradient, step_size)  # Compute the update step
        x = x + update  # Update the current point

        # Store the function value for plotting/convergence analysis
        history.append(f(x))

        # Check for convergence (stop if the gradient is very small)
        if np.linalg.norm(gradient) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    return x, history


# ## Rms prop algo 


import numpy as np

def rmsprop_update(gradient, alpha, beta, epsilon=1e-8):
    """
    Compute the update step using RMSProp.

    Parameters:
    - gradient: The gradient at the current point.
    - alpha: Learning rate.
    - beta: Decay rate for the moving average of squared gradients.
    - epsilon: Small constant to avoid division by zero.

    Returns:
    - update: The update step.
    - cache: Updated moving average of squared gradients.
    """
    # Initialize cache (moving average of squared gradients)
    if not hasattr(rmsprop_update, 'cache'):
        rmsprop_update.cache = np.zeros_like(gradient)
    
    # Update cache with squared gradients
    rmsprop_update.cache = beta * rmsprop_update.cache + (1 - beta) * gradient**2
    
    # Compute the update step
    update = -alpha * gradient / (np.sqrt(rmsprop_update.cache) + epsilon)
    return update, rmsprop_update.cache

def gradient_descent_rmsprop(f, grad_f, x0, alpha, beta, max_iterations=1000, tol=1e-6):
    """
    Perform gradient descent using RMSProp.

    Parameters:
    - f: The function to minimize.
    - grad_f: The gradient of the function.
    - x0: Initial point (numpy array).
    - alpha: Learning rate.
    - beta: Decay rate for the moving average of squared gradients.
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.

    Returns:
    - x: The final point after optimization.
    - history: List of function values at each iteration.
    """
    x = x0  # Initialize the current point
    history = []  # Store the function values at each iteration

    for iteration in range(max_iterations):
        gradient = grad_f(x)  # Compute the gradient at the current point
        update, _ = rmsprop_update(gradient, alpha, beta)  # Compute the update step
        x = x + update  # Update the current point

        # Store the function value for plotting/convergence analysis
        history.append(f(x))

        # Check for convergence (stop if the gradient is very small)
        if np.linalg.norm(gradient) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    return x, history


# ## heavy ball/momentum algo


import numpy as np

def heavy_ball_update(gradient, alpha, beta):
    """
    Compute the update step using Heavy Ball (Momentum).

    Parameters:
    - gradient: The gradient at the current point.
    - alpha: Learning rate.
    - beta: Momentum term (fraction of the previous update).

    Returns:
    - update: The update step.
    - velocity: Updated velocity (momentum term).
    """
    # Initialize velocity (momentum term)
    if not hasattr(heavy_ball_update, 'velocity'):
        heavy_ball_update.velocity = np.zeros_like(gradient)
    
    # Update velocity
    heavy_ball_update.velocity = beta * heavy_ball_update.velocity - alpha * gradient
    
    # Compute the update step
    update = heavy_ball_update.velocity
    return update, heavy_ball_update.velocity

def gradient_descent_heavy_ball(f, grad_f, x0, alpha, beta, max_iterations=1000, tol=1e-6):
    """
    Perform gradient descent using Heavy Ball (Momentum).

    Parameters:
    - f: The function to minimize.
    - grad_f: The gradient of the function.
    - x0: Initial point (numpy array).
    - alpha: Learning rate.
    - beta: Momentum term (fraction of the previous update).
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.

    Returns:
    - x: The final point after optimization.
    - history: List of function values at each iteration.
    """
    x = x0  # Initialize the current point
    history = []  # Store the function values at each iteration

    for iteration in range(max_iterations):
        gradient = grad_f(x)  # Compute the gradient at the current point
        update, _ = heavy_ball_update(gradient, alpha, beta)  # Compute the update step
        x = x + update  # Update the current point

        # Store the function value for plotting/convergence analysis
        history.append(f(x))

        # Check for convergence (stop if the gradient is very small)
        if np.linalg.norm(gradient) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    return x, history


# ### adam algo


import numpy as np

def adam_update(gradient, alpha, beta1, beta2, epsilon=1e-8, iteration=1):
    """
    Compute the update step using Adam.

    Parameters:
    - gradient: The gradient at the current point.
    - alpha: Learning rate.
    - beta1: Decay rate for the moving average of gradients.
    - beta2: Decay rate for the moving average of squared gradients.
    - epsilon: Small constant to avoid division by zero.
    - iteration: Current iteration number (for bias correction).

    Returns:
    - update: The update step.
    - m: Updated moving average of gradients.
    - v: Updated moving average of squared gradients.
    """
    # Initialize moving averages
    if not hasattr(adam_update, 'm'):
        adam_update.m = np.zeros_like(gradient)
        adam_update.v = np.zeros_like(gradient)
    
    # Update moving averages
    adam_update.m = beta1 * adam_update.m + (1 - beta1) * gradient
    adam_update.v = beta2 * adam_update.v + (1 - beta2) * gradient**2
    
    # Bias correction
    m_hat = adam_update.m / (1 - beta1**iteration)
    v_hat = adam_update.v / (1 - beta2**iteration)
    
    # Compute the update step
    update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, adam_update.m, adam_update.v

def gradient_descent_adam(f, grad_f, x0, alpha, beta1, beta2, max_iterations=1000, tol=1e-6):
    """
    Perform gradient descent using Adam.

    Parameters:
    - f: The function to minimize.
    - grad_f: The gradient of the function.
    - x0: Initial point (numpy array).
    - alpha: Learning rate.
    - beta1: Decay rate for the moving average of gradients.
    - beta2: Decay rate for the moving average of squared gradients.
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.

    Returns:
    - x: The final point after optimization.
    - history: List of function values at each iteration.
    """
    x = x0  # Initialize the current point
    history = []  # Store the function values at each iteration

    for iteration in range(1, max_iterations + 1):
        gradient = grad_f(x)  # Compute the gradient at the current point
        update, _, _ = adam_update(gradient, alpha, beta1, beta2, iteration=iteration)  # Compute the update step
        x = x + update  # Update the current point

        # Store the function value for plotting/convergence analysis
        history.append(f(x))

        # Check for convergence (stop if the gradient is very small)
        if np.linalg.norm(gradient) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    return x, history


# ## Part b 1 rms prop 


# ### RMSProp Implementation


import numpy as np
import matplotlib.pyplot as plt

# Function 1: f1(x, y) = 6(x-1)^4 + 8(y-2)^2
def f1(x):
    return 6 * (x[0] - 1) ** 4 + 8 * (x[1] - 2) ** 2

def grad_f1(x):
    return np.array([24 * (x[0] - 1) ** 3, 16 * (x[1] - 2)])

# Function 2: f2(x, y) = max(x-1,0) + 8|y-2|
def f2(x):
    return np.maximum(x[0] - 1, 0) + 8 * np.abs(x[1] - 2)

def grad_f2(x):
    return np.array([
        1.0 if x[0] > 1 else 0.0,  # Derivative of max(x-1, 0)
        8 * np.sign(x[1] - 2)  # Derivative of 8|y-2|
    ])

# RMSProp Update Rule
def rmsprop_update(gradient, cache, alpha, beta, epsilon=1e-8):
    cache = beta * cache + (1 - beta) * gradient**2
    update = -alpha * gradient / (np.sqrt(cache) + epsilon)
    return update, cache

# RMSProp Gradient Descent Function
def gradient_descent_rmsprop(f, grad_f, x0, alpha, beta, max_iterations=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = []
    cache = np.zeros_like(x)  # Initialize cache

    for iteration in range(max_iterations):
        gradient = grad_f(x)
        update, cache = rmsprop_update(gradient, cache, alpha, beta)
        x += update
        history.append(f(x))

        # Print values every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: f(x, y) = {f(x):.4f}, x = {x[0]:.4f}, y = {x[1]:.4f}")

        # Stop if gradient norm is small
        if np.linalg.norm(gradient) < tol:
            print(f"Converged at Iteration {iteration}: f(x, y) = {f(x):.4f}, x = {x[0]:.4f}, y = {x[1]:.4f}")
            break

    return x, history

# Parameters for RMSProp
alphas = [0.01, 0.05, 0.1]  # Learning rates to test
betas = [0.25, 0.9]  # Decay rates
x0_f1 = [0.0, 0.0]  # Initial point for f1
x0_f2 = [0.5, 3.0]  # Initial point for f2

# Run RMSProp and plot results for Function 1 and Function 2
functions = [f1, f2]
grad_functions = [grad_f1, grad_f2]
initial_points = [x0_f1, x0_f2]
function_names = ["Function 1", "Function 2"]

for i, (f, grad_f, x0, name) in enumerate(zip(functions, grad_functions, initial_points, function_names)):
    plt.figure(figsize=(12, 6))
    for beta in betas:
        for alpha in alphas:
            _, history = gradient_descent_rmsprop(f, grad_f, x0, alpha, beta)
            plt.plot(history, label=f'α={alpha}, β={beta}')
    
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(f"RMSProp Convergence for {name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
# Generate Contour Plots with Trajectory
def plot_contour(f, x_history, y_history, name):
    x_range = np.linspace(-1, 3, 400)
    y_range = np.linspace(0, 5, 400)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    
    # Compute function values over the grid
    if name == "Function 1":
        z_mesh = 6 * (x_mesh - 1) ** 4 + 8 * (y_mesh - 2) ** 2
    else:
        z_mesh = np.maximum(x_mesh - 1, 0) + 8 * np.abs(y_mesh - 2)

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(x_mesh, y_mesh, z_mesh, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(cp)
    plt.plot(x_history, y_history, 'ro-', label='Optimizer Path')  # Plot trajectory
    plt.title(f'Contour Plot of {name} with RMSProp Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


for i, (f, grad_f, x0, name) in enumerate(zip(functions, grad_functions, initial_points, function_names)):
    for beta in betas:
        for alpha in alphas:
            x_history, y_history = [], []
            x = np.array(x0, dtype=float)
            cache = np.zeros_like(x)  # Reset cache
            
            for iteration in range(500):  # Limit iterations for better visualization
                gradient = grad_f(x)
                update, cache = rmsprop_update(gradient, cache, alpha, beta)
                x += update
                x_history.append(x[0])
                y_history.append(x[1])

            # Plot Contour
            plot_contour(f, x_history, y_history, name)


# ### part b 2 heavy ball


import numpy as np
import matplotlib.pyplot as plt

# Function 1: f1(x, y) = 6(x-1)^4 + 8(y-2)^2
def f1(x):
    return 6 * (x[0] - 1) ** 4 + 8 * (x[1] - 2) ** 2

def grad_f1(x):
    return np.array([24 * (x[0] - 1) ** 3, 16 * (x[1] - 2)])

# Function 2: f2(x, y) = max(x-1,0) + 8|y-2|
def f2(x):
    return np.maximum(x[0] - 1, 0) + 8 * np.abs(x[1] - 2)

def grad_f2(x):
    return np.array([
        1.0 if x[0] > 1 else 0.0,
        8 * np.sign(x[1] - 2)
    ])

# Heavy Ball Update Rule
def heavy_ball_update(grad, velocity, alpha, beta):
    velocity = beta * velocity - alpha * grad
    return velocity, velocity

# Heavy Ball Gradient Descent with Improved Convergence Checking
def gradient_descent_heavy_ball(f, grad_f, x0, alpha, beta, max_iterations=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = []
    velocity = np.zeros_like(x)  # Initialize velocity
    f_prev = f(x)  # Store previous function value

    for iteration in range(max_iterations):
        grad = grad_f(x)
        update, velocity = heavy_ball_update(grad, velocity, alpha, beta)
        x += update
        f_current = f(x)
        history.append(f_current)

        # Print values every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: f(x, y) = {f_current:.4f}, x = {x[0]:.4f}, y = {x[1]:.4f}")

        # Stop if gradient norm is small OR function value stops decreasing
        if np.linalg.norm(grad) < tol or abs(f_current - f_prev) < 1e-8:
            print(f"Converged at Iteration {iteration}: f(x, y) = {f_current:.4f}, x = {x[0]:.4f}, y = {x[1]:.4f}")
            break

        f_prev = f_current  # Update previous function value

    return x, history

# Parameters for Heavy Ball with Reduced Momentum for Function 2
alphas = [0.001, 0.005, 0.01]  # Learning rates
betas_f1 = [0.25, 0.9]  # Momentum factors for Function 1
betas_f2 = [0.1, 0.5]  # Reduced momentum for Function 2 to prevent oscillations
x0_f1 = [0.0, 0.0]  # Initial point for f1
x0_f2 = [0.5, 3.0]  # Initial point for f2

# Run Heavy Ball and plot results for Function 1
plt.figure(figsize=(12, 6))
for beta in betas_f1:
    for alpha in alphas:
        _, history = gradient_descent_heavy_ball(f1, grad_f1, x0_f1, alpha, beta)
        plt.plot(history, label=f'α={alpha}, β={beta}')
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Heavy Ball Convergence for Function 1")
plt.legend()
plt.grid(True)
plt.show()

# Run Heavy Ball and plot results for Function 2 (Reduced Momentum)
plt.figure(figsize=(12, 6))
for beta in betas_f2:  # Use smaller momentum for stability
    for alpha in alphas:
        _, history = gradient_descent_heavy_ball(f2, grad_f2, x0_f2, alpha, beta)
        plt.plot(history, label=f'α={alpha}, β={beta}')
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Heavy Ball Convergence for Function 2 (Adjusted β)")
plt.legend()
plt.grid(True)
plt.show()

# # contoutr plot
# Generate Contour Plots with Trajectory
def plot_contour(f, x_history, y_history, name):
    x_range = np.linspace(-1, 3, 400)
    y_range = np.linspace(0, 5, 400)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    
    # Compute function values over the grid
    if name == "Function 1":
        z_mesh = 6 * (x_mesh - 1) ** 4 + 8 * (y_mesh - 2) ** 2
    else:
        z_mesh = np.maximum(x_mesh - 1, 0) + 8 * np.abs(y_mesh - 2)

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(x_mesh, y_mesh, z_mesh, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(cp)
    plt.plot(x_history, y_history, 'ro-', label='Optimizer Path')  # Plot trajectory
    plt.title(f'Contour Plot of {name} with Heavy Ball Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Run Heavy Ball for contour plotting and collect history
for i, (f, grad_f, x0, name) in enumerate(zip([f1, f2], [grad_f1, grad_f2], [[0, 0], [0.5, 3.0]], ["Function 1", "Function 2"])):
    for beta in [0.25, 0.9]:  # Using both momentum values
        for alpha in [0.001, 0.005, 0.01]:  
            x_history, y_history = [], []
            x = np.array(x0, dtype=float)
            velocity = np.zeros_like(x)  # Reset velocity
            
            for iteration in range(300):  # Limit iterations for better visualization
                grad = grad_f(x)
                update, velocity = heavy_ball_update(grad, velocity, alpha, beta)
                x += update
                x_history.append(x[0])
                y_history.append(x[1])

            # Plot Contour
            plot_contour(f, x_history, y_history, name)

# # Step Size vs Iteration
def plot_step_size(f, grad_f, x0, alpha, beta, name):
    x = np.array(x0, dtype=float)
    velocity = np.zeros_like(x)
    step_sizes = []
    
    for iteration in range(1000):
        grad = grad_f(x)
        update, velocity = heavy_ball_update(grad, velocity, alpha, beta)
        x += update
        step_sizes.append(np.linalg.norm(update))  # Compute step size
        
    plt.plot(step_sizes, label=f'α={alpha}, β={beta}')

for i, (f, grad_f, x0, name) in enumerate(zip([f1, f2], [grad_f1, grad_f2], [[0, 0], [0.5, 3.0]], ["Function 1", "Function 2"])):
    plt.figure(figsize=(10, 6))
    for beta in [0.25, 0.9]:  # Using both momentum values
        for alpha in [0.001, 0.005, 0.01]:  
            plot_step_size(f, grad_f, x0, alpha, beta, name)

    plt.xlabel("Iteration")
    plt.ylabel("Step Size (Update Magnitude)")
    plt.title(f"Step Size vs Iteration for {name}")
    plt.legend()
    plt.grid(True)
    plt.show()



# ## adam


import numpy as np
import matplotlib.pyplot as plt

# Adam Update Rule
def adam_update(grad, m, v, t, alpha, beta1, beta2, epsilon=1e-8):
    """Performs an Adam optimization update step."""
    m = beta1 * m + (1 - beta1) * grad  # First moment estimate
    v = beta2 * v + (1 - beta2) * (grad ** 2)  # Second moment estimate

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)  # Compute update step
    return update, m, v  # Return update, new moment estimates

# Adam Gradient Descent Implementation
def gradient_descent_adam(f, grad_f, x0, alpha, beta1, beta2, max_iterations=2000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = []
    m, v = np.zeros_like(x), np.zeros_like(x)  # Initialize moment estimates
    f_prev = f(x)  # Store previous function value

    for t in range(1, max_iterations + 1):
        grad = grad_f(x)
        update, m, v = adam_update(grad, m, v, t, alpha, beta1, beta2)
        x += update
        f_current = f(x)
        history.append(f_current)

        # Print values every 50 iterations
        if t % 50 == 0:
            print(f"Iteration {t}: f(x, y) = {f_current:.4f}, x = {x[0]:.4f}, y = {x[1]:.4f}")

        # Stop if gradient norm is small OR function value stops decreasing
        if np.linalg.norm(grad) < 1e-4 or abs(f_current - f_prev) < 1e-6:
            print(f"Converged at Iteration {t}: f(x, y) = {f_current:.4f}, x = {x[0]:.4f}, y = {x[1]:.4f}")
            break

        f_prev = f_current  # Update previous function value

    return x, history

# Parameters for Adam
alphas = [0.001, 0.005, 0.01]  # Learning rates
beta1_values = [0.9, 0.95]  # First moment decay rates
beta2_values = [0.99, 0.999]  # Second moment decay rates
x0_f1 = [0.0, 0.0]  # Initial point for f1
x0_f2 = [0.5, 3.0]  # Initial point for f2

# Run Adam and plot results for Function 1
plt.figure(figsize=(12, 6))
for beta1 in beta1_values:
    for beta2 in beta2_values:
        for alpha in alphas:
            _, history = gradient_descent_adam(f1, grad_f1, x0_f1, alpha, beta1, beta2)
            plt.plot(history, label=f'α={alpha}, β1={beta1}, β2={beta2}')
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Adam Convergence for Function 1")
plt.legend()
plt.grid(True)
plt.show()

# Run Adam and plot results for Function 2
plt.figure(figsize=(12, 6))
for beta1 in beta1_values:
    for beta2 in beta2_values:
        for alpha in alphas:
            _, history = gradient_descent_adam(f2, grad_f2, x0_f2, alpha, beta1, beta2)
            plt.plot(history, label=f'α={alpha}, β1={beta1}, β2={beta2}')
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Adam Convergence for Function 2")
plt.legend()
plt.grid(True)
plt.show()


# ## contour plot 
# Generate Contour Plots with Trajectory
def plot_contour_adam(f, x_history, y_history, name):
    x_range = np.linspace(-1, 3, 400)
    y_range = np.linspace(0, 5, 400)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    
    # Compute function values over the grid
    if name == "Function 1":
        z_mesh = 6 * (x_mesh - 1) ** 4 + 8 * (y_mesh - 2) ** 2
    else:
        z_mesh = np.maximum(x_mesh - 1, 0) + 8 * np.abs(y_mesh - 2)

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(x_mesh, y_mesh, z_mesh, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(cp)
    plt.plot(x_history, y_history, 'ro-', label='Optimizer Path')  # Plot trajectory
    plt.title(f'Contour Plot of {name} with Adam Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Run Adam for contour plotting and collect history
for i, (f, grad_f, x0, name) in enumerate(zip([f1, f2], [grad_f1, grad_f2], [[0, 0], [0.5, 3.0]], ["Function 1", "Function 2"])):
    for beta1 in beta1_values:
        for beta2 in beta2_values:
            for alpha in alphas:
                x_history, y_history = [], []
                x = np.array(x0, dtype=float)
                m, v = np.zeros_like(x), np.zeros_like(x)

                for t in range(300):  # Limit iterations for better visualization
                    grad = grad_f(x)
                    update, m, v = adam_update(grad, m, v, t + 1, alpha, beta1, beta2)
                    x += update
                    x_history.append(x[0])
                    y_history.append(x[1])

                # Plot Contour
                plot_contour_adam(f, x_history, y_history, name)


# ### Part c (i)

### Implementing the ReLU Function and its Derivative
import numpy as np
import matplotlib.pyplot as plt

# Define ReLU Function
def relu(x):
    return np.maximum(0, x)

# Define ReLU Gradient using Heaviside function
def grad_relu(x):
    return np.heaviside(x, 0)  # 0 for x < 0, 1 for x > 0

### Apply RMSProp to ReLU

# RMSProp Update Function
def rmsprop_update(grad, s, alpha, beta, epsilon=1e-8):
    s = beta * s + (1 - beta) * grad**2
    update = -alpha * grad / (np.sqrt(s) + epsilon)
    return update, s

# Apply RMSProp to ReLU
def rmsprop_relu(x0, alpha=0.01, beta=0.9, max_iterations=100):
    x = x0
    s = 0  # Initialize moving average
    history = []

    for iteration in range(max_iterations):
        grad = grad_relu(x)
        update, s = rmsprop_update(grad, s, alpha, beta)
        x += update
        history.append(x)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: x = {x:.4f}")

        # If gradient is 0 (ReLU is flat), stop updating
        if grad == 0:
            print(f"Converged at Iteration {iteration}: x = {x:.4f}")
            break

    return history

# Run RMSProp for different initial conditions
x_vals = [-1, 1, 100]
plt.figure(figsize=(8, 6))
for x0 in x_vals:
    history = rmsprop_relu(x0)
    plt.plot(history, label=f"Initial x={x0}")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.title("RMSProp Optimization on ReLU")
plt.legend()
plt.grid()
plt.show()


# ### Apply Heavy Ball to ReLU

# %%
# Heavy Ball Update Function
def heavy_ball_update(grad, velocity, alpha, beta):
    velocity = beta * velocity - alpha * grad
    update = velocity
    return update, velocity

# Apply Heavy Ball to ReLU
def heavy_ball_relu(x0, alpha=0.01, beta=0.9, max_iterations=100):
    x = x0
    velocity = 0  # Initialize velocity
    history = []

    for iteration in range(max_iterations):
        grad = grad_relu(x)
        update, velocity = heavy_ball_update(grad, velocity, alpha, beta)
        x += update
        history.append(x)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: x = {x:.4f}")

        if grad == 0:
            print(f"Converged at Iteration {iteration}: x = {x:.4f}")
            break

    return history


plt.figure(figsize=(8, 6))
for x0 in x_vals:
    history = heavy_ball_relu(x0)
    plt.plot(history, label=f"Initial x={x0}")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.title("Heavy Ball Optimization on ReLU")
plt.legend()
plt.grid()
plt.show()

### Apply Adam to ReLU

# Adam Update Function
def adam_update(grad, m, v, t, alpha, beta1, beta2, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return update, m, v

# Apply Adam to ReLU
def adam_relu(x0, alpha=0.01, beta1=0.9, beta2=0.99, max_iterations=100):
    x = x0
    m, v = 0, 0
    history = []

    for t in range(1, max_iterations + 1):
        grad = grad_relu(x)
        update, m, v = adam_update(grad, m, v, t, alpha, beta1, beta2)
        x += update
        history.append(x)

        if t % 10 == 0:
            print(f"Iteration {t}: x = {x:.4f}")

        if grad == 0:
            print(f"Converged at Iteration {t}: x = {x:.4f}")
            break

    return history
plt.figure(figsize=(8, 6))
for x0 in x_vals:
    history = adam_relu(x0)
    plt.plot(history, label=f"Initial x={x0}")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.title("Adam Optimization on ReLU")
plt.legend()
plt.grid()
plt.show()





