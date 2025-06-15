
# ## Part A (i)

# %%
import sympy as sp

# Define the symbolic variable
x = sp.symbols('x')

# Define the function y(x) = x^4
y = x**4

# Compute the derivative dy/dx
dy_dx = sp.diff(y, x)

# Display the function and its derivative
print("Function: y =", y)
print("Derivative: dy/dx =", dy_dx)


# %% [markdown]
# ## Part A (ii)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x_values = np.linspace(-2, 2, 400)

# Calculate exact derivatives using the expression derived
exact_derivatives = 4 * x_values**3

# Calculate finite difference derivatives
delta = 0.01
finite_diff_derivatives = (x_values**4 + delta - (x_values - delta)**4) / (2 * delta)

# Plotting both sets of derivatives
plt.figure(figsize=(10, 6))
plt.plot(x_values, exact_derivatives, label='Exact Derivative $4x^3$', color='blue')
plt.plot(x_values, finite_diff_derivatives, label='Finite Difference Approximation', color='red', linestyle='--')
plt.title('Comparison of Exact and Finite Difference Derivatives')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.legend()
plt.grid(True)
plt.show()


# ## Part A (iii)
# Define a range of delta values
delta_values = np.array([0.001, 0.01, 0.1, 1])

# Prepare the plot
plt.figure(figsize=(12, 8))

# Compute and plot the errors for each delta
for delta in delta_values:
    finite_diff_derivatives_delta = (x_values**4 + delta - (x_values - delta)**4) / (2 * delta)
    error = np.abs(finite_diff_derivatives_delta - exact_derivatives)
    plt.plot(x_values, error, label=f'Error for δ = {delta}')

# Enhance plot readability
plt.title('Error in Finite Difference Derivatives vs. Exact Derivatives for Various δ')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Log scale for better visibility of errors across scales
plt.show()



# ### part B (i)

def gradient_descent(x_initial, alpha, num_iterations):
    x = x_initial
    history = []  # To store the history of x and y values

    # Function y = x^4 and its derivative
    y = lambda x: x**4
    dy_dx = lambda x: 4 * x**3

    for _ in range(num_iterations):
        current_y = y(x)
        gradient = dy_dx(x)
        x = x - alpha * gradient  # Update x by taking a step in the direction of the steepest descent
        history.append((x, current_y))

    return history

# Parameters
x_initial = 1.0
alpha = 0.01  # Step size
num_iterations = 50

# Run gradient descent
history = gradient_descent(x_initial, alpha, num_iterations)

# Display the first few steps
history[:10]  # Show only the first 10 iterations for brevity


# ## Part B (ii)

# %%
def gradient_descent_plot(x_initial, alpha, num_iterations):
    x = x_initial
    x_history = []  # To store the history of x values
    y_history = []  # To store the history of y values

    # Function y = x^4 and its derivative
    y = lambda x: x**4
    dy_dx = lambda x: 4 * x**3

    for _ in range(num_iterations):
        current_y = y(x)
        x_history.append(x)
        y_history.append(current_y)
        gradient = dy_dx(x)
        x = x - alpha * gradient  # Update x by taking a step in the direction of the steepest descent

    return x_history, y_history

# Parameters for this part
alpha = 0.1  # Step size
num_iterations = 20  # Using fewer iterations for clarity in visualization

# Run gradient descent
x_history, y_history = gradient_descent_plot(x_initial, alpha, num_iterations)

# Plot for the values of x across iterations
plt.figure(figsize=(8, 6))
plt.plot(x_history, marker='o')
plt.title('Values of x across iterations')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.grid(True)
plt.show()

# Plot for the values of y(x) across iterations
plt.figure(figsize=(8, 6))
plt.plot(y_history, marker='o', color='r')
plt.title('Values of y(x) across iterations')
plt.xlabel('Iteration')
plt.ylabel('y(x) value')
plt.grid(True)
plt.show()


# ### part b (iii)

# %%
# Define a range of initial x values and step sizes
initial_x_values = np.linspace(-1.5, 1.5, 7)  # A range of initial values around zero
step_sizes = [0.01, 0.05, 0.1, 0.2]  # Different step sizes
num_iterations = 50  # Consistent iteration count for comparison

# Prepare to store results for plotting
results = []

# Run gradient descent for each combination of initial x and alpha
for x_initial in initial_x_values:
    for alpha in step_sizes:
        x_history, _ = gradient_descent_plot(x_initial, alpha, num_iterations)
        results.append((x_initial, alpha, x_history[-1]))

# Plotting the results
plt.figure(figsize=(12, 8))
for i, alpha in enumerate(step_sizes):
    x_final_values = [result[2] for result in results if result[1] == alpha]
    plt.plot(initial_x_values, x_final_values, marker='o', label=f'α = {alpha}')

plt.title('Final x values after 50 iterations for different α and initial x')
plt.xlabel('Initial x value')
plt.ylabel('Final x value')
plt.legend()
plt.grid(True)
plt.show()


# ### part c (i)

def gradient_descent_gamma(x_initial, alpha, gamma, num_iterations):
    x = x_initial
    x_history = []  # To store the history of x values
    y_history = []  # To store the history of y values

    # Function y = γ * x^2 and its derivative
    y = lambda x: gamma * x**2
    dy_dx = lambda x: 2 * gamma * x

    for _ in range(num_iterations):
        current_y = y(x)
        x_history.append(x)
        y_history.append(current_y)
        gradient = dy_dx(x)
        x = x - alpha * gradient  # Update x by taking a step in the direction of the steepest descent

    return x_history, y_history

# Parameters for this experiment
gamma_values = [0.1, 1, 10]  # Different gamma values
alpha = 0.1  # Fixed step size
num_iterations = 50
x_initial = 1.0  # Fixed initial x value

# Run gradient descent for different gamma values and plot the results
plt.figure(figsize=(12, 6))
for gamma in gamma_values:
    _, y_history = gradient_descent_gamma(x_initial, alpha, gamma, num_iterations)
    plt.plot(y_history, marker='o', label=f'γ = {gamma}')

plt.title('Convergence of y(x) = γx² for Different γ Values')
plt.xlabel('Iteration')
plt.ylabel('y(x) Value')
plt.legend()
plt.grid(True)
plt.show()

# ### part c (ii)

def gradient_descent_abs_gamma(x_initial, alpha, gamma, num_iterations):
    x = x_initial
    x_history = []  # To store the history of x values
    y_history = []  # To store the history of y values

    # Function y = γ * |x| and its subgradient
    y = lambda x: gamma * abs(x)
    dy_dx = lambda x: gamma * np.sign(x) if x != 0 else 0

    for _ in range(num_iterations):
        current_y = y(x)
        x_history.append(x)
        y_history.append(current_y)
        gradient = dy_dx(x)
        x = x - alpha * gradient  # Update x by taking a step in the direction of the steepest descent

    return x_history, y_history

# Parameters for this experiment
gamma_values = [0.1, 1, 10]  # Different gamma values
alpha = 0.1  # Fixed step size
num_iterations = 50
x_initial = 1.0  # Fixed initial x value

# Run gradient descent for different gamma values and plot the results
plt.figure(figsize=(12, 6))
for gamma in gamma_values:
    _, y_history = gradient_descent_abs_gamma(x_initial, alpha, gamma, num_iterations)
    plt.plot(y_history, marker='o', label=f'γ = {gamma}')

plt.title('Convergence of y(x) = γ|x| for Different γ Values')
plt.xlabel('Iteration')
plt.ylabel('y(x) Value')
plt.legend()
plt.grid(True)
plt.show()





