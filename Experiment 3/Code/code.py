# %%
### Function that is given

# %% [markdown]
# import numpy as np
# 
# def generate_trainingdata(m=25):
#     return np.array([0,0])+0.25*np.random.randn(m,2)
# 
# def f(x, minibatch):
#     # loss function sum_{w in training data} f(x,w)
#     y=0; count=0
#     for w in minibatch:
#         z=x-w-1
#         y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
#         count=count+1
#     return y/count

# %%
### a(i) Mini Batch SGD Implementation

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loss function provided in the assignment
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
        count=count+1
    return y/count

# Finite difference gradient calculation
def gradient_f(x, minibatch, h=1e-6):
    """Calculate gradient of f using finite differences."""
    grad = np.zeros(2)
    for i in range(2):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (f(x_plus_h, minibatch) - f(x, minibatch)) / h
    return grad

class SGDOptimizer:
    def __init__(self, initial_x, training_data):
        """
        Initialize the SGD optimizer.
        
        Parameters:
        - initial_x: Starting point (numpy array of shape (2,))
        - training_data: Full training dataset
        """
        self.x = initial_x.copy()
        self.training_data = training_data
        
        # For tracking progress
        self.losses = []
        self.trajectory = [initial_x.copy()]
    
    def _get_mini_batch(self, batch_size):
        """Randomly select a mini-batch from the training data."""
        indices = np.random.choice(len(self.training_data), batch_size, replace=False)
        return self.training_data[indices]
    
    def constant_step(self, step_size, batch_size, max_iterations=100):
        """
        Mini-batch SGD with constant step size.
        
        Parameters:
        - step_size: Learning rate
        - batch_size: Size of mini-batches
        - max_iterations: Maximum number of iterations
        
        Returns:
        - x: Final position
        - losses: List of loss values at each iteration
        - trajectory: List of positions at each iteration
        """
        self.x = self.trajectory[0].copy()  # Reset to initial position
        self.losses = [f(self.x, self.training_data)]
        self.trajectory = [self.x.copy()]
        
        for _ in range(max_iterations):
            # Get mini-batch
            mini_batch = self._get_mini_batch(batch_size)
            
            # Calculate gradient
            grad = gradient_f(self.x, mini_batch)
            
            # Update x
            self.x = self.x - step_size * grad
            
            # Store loss and trajectory
            self.losses.append(f(self.x, self.training_data))
            self.trajectory.append(self.x.copy())
        
        return self.x, self.losses, self.trajectory
    
    def polyak_step(self, batch_size, f_star, max_step_size=0.1, max_iterations=100):
        """
        Mini-batch SGD with Polyak step size.
        
        Parameters:
        - batch_size: Size of mini-batches
        - f_star: Optimal function value
        - max_step_size: Maximum allowed step size (for stability)
        - max_iterations: Maximum number of iterations
        
        Returns:
        - x: Final position
        - losses: List of loss values at each iteration
        - trajectory: List of positions at each iteration
        """
        self.x = self.trajectory[0].copy()  # Reset to initial position
        self.losses = [f(self.x, self.training_data)]
        self.trajectory = [self.x.copy()]
        
        for _ in range(max_iterations):
            # Get mini-batch
            mini_batch = self._get_mini_batch(batch_size)
            
            # Calculate gradient
            grad = gradient_f(self.x, mini_batch)
            
            # Calculate current loss on mini-batch
            current_f = f(self.x, mini_batch)
            
            # Calculate Polyak step size
            grad_norm_squared = np.sum(grad**2)
            if grad_norm_squared < 1e-10 or current_f <= f_star:
                step = 0  # No step if gradient is very small or already at optimum
            else:
                step = min((current_f - f_star) / grad_norm_squared, max_step_size)
            
            # Update x
            self.x = self.x - step * grad
            
            # Store loss and trajectory
            self.losses.append(f(self.x, self.training_data))
            self.trajectory.append(self.x.copy())
        
        return self.x, self.losses, self.trajectory
    
    def rmsprop(self, step_size, batch_size, beta=0.9, epsilon=1e-8, max_iterations=100):
        """
        Mini-batch SGD with RMSProp.
        
        Parameters:
        - step_size: Base learning rate
        - batch_size: Size of mini-batches
        - beta: Decay rate for moving average
        - epsilon: Small constant to avoid division by zero
        - max_iterations: Maximum number of iterations
        
        Returns:
        - x: Final position
        - losses: List of loss values at each iteration
        - trajectory: List of positions at each iteration
        """
        self.x = self.trajectory[0].copy()  # Reset to initial position
        self.losses = [f(self.x, self.training_data)]
        self.trajectory = [self.x.copy()]
        
        # Initialize squared gradient accumulator
        s = np.zeros_like(self.x)
        
        for _ in range(max_iterations):
            # Get mini-batch
            mini_batch = self._get_mini_batch(batch_size)
            
            # Calculate gradient
            grad = gradient_f(self.x, mini_batch)
            
            # Update squared gradient accumulator
            s = beta * s + (1 - beta) * grad**2
            
            # Calculate adaptive step size
            adjusted_step = step_size / (np.sqrt(s) + epsilon)
            
            # Update x
            self.x = self.x - adjusted_step * grad
            
            # Store loss and trajectory
            self.losses.append(f(self.x, self.training_data))
            self.trajectory.append(self.x.copy())
        
        return self.x, self.losses, self.trajectory
    
    def heavy_ball(self, step_size, batch_size, beta=0.9, max_iterations=100):
        """
        Mini-batch SGD with Heavy Ball (momentum).
        
        Parameters:
        - step_size: Learning rate
        - batch_size: Size of mini-batches
        - beta: Momentum parameter
        - max_iterations: Maximum number of iterations
        
        Returns:
        - x: Final position
        - losses: List of loss values at each iteration
        - trajectory: List of positions at each iteration
        """
        self.x = self.trajectory[0].copy()  # Reset to initial position
        self.losses = [f(self.x, self.training_data)]
        self.trajectory = [self.x.copy()]
        
        # Initialize velocity
        v = np.zeros_like(self.x)
        
        for _ in range(max_iterations):
            # Get mini-batch
            mini_batch = self._get_mini_batch(batch_size)
            
            # Calculate gradient
            grad = gradient_f(self.x, mini_batch)
            
            # Update with momentum
            v = beta * v - step_size * grad
            self.x = self.x + v
            
            # Store loss and trajectory
            self.losses.append(f(self.x, self.training_data))
            self.trajectory.append(self.x.copy())
        
        return self.x, self.losses, self.trajectory
    
    def adam(self, step_size, batch_size, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=100):
        """
        Mini-batch SGD with Adam optimizer.
        
        Parameters:
        - step_size: Base learning rate
        - batch_size: Size of mini-batches
        - beta1: Exponential decay rate for first moment
        - beta2: Exponential decay rate for second moment
        - epsilon: Small constant to avoid division by zero
        - max_iterations: Maximum number of iterations
        
        Returns:
        - x: Final position
        - losses: List of loss values at each iteration
        - trajectory: List of positions at each iteration
        """
        self.x = self.trajectory[0].copy()  # Reset to initial position
        self.losses = [f(self.x, self.training_data)]
        self.trajectory = [self.x.copy()]
        
        # Initialize moment estimates
        m = np.zeros_like(self.x)  # First moment
        v = np.zeros_like(self.x)  # Second moment
        t = 0  # Timestep
        
        for _ in range(max_iterations):
            t += 1
            
            # Get mini-batch
            mini_batch = self._get_mini_batch(batch_size)
            
            # Calculate gradient
            grad = gradient_f(self.x, mini_batch)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (grad**2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2**t)
            
            # Update parameters
            self.x = self.x - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Store loss and trajectory
            self.losses.append(f(self.x, self.training_data))
            self.trajectory.append(self.x.copy())
        
        return self.x, self.losses, self.trajectory

# Visualization functions
def plot_loss_surface(training_data, x_range=(-5, 15), y_range=(-5, 5), resolution=100):
    """Plot the loss function surface and contour."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), training_data)
    
    return X, Y, Z

def plot_wireframe(X, Y, Z, title='Loss Function Surface'):
    """Plot a 3D wireframe of the loss function."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('Loss')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig

def plot_contour(X, Y, Z, trajectory=None, title='Loss Function Contour'):
    """Plot a contour map of the loss function with optional trajectory."""
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    # Add trajectory if provided
    if trajectory is not None:
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, 'r.-', linewidth=2, markersize=10, label='Optimization Path')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start Point')
        plt.plot(traj_x[-1], traj_y[-1], 'bo', markersize=10, label='End Point')
        plt.legend()
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(title)
    return fig

def plot_loss_vs_iterations(losses, title='Loss vs. Iterations'):
    """Plot how the loss changes over iterations."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    return fig

# %%
#### a(ii) (2 plots) (loss_surface_wireframe, loss_surface_contour.png)

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Functions from previous code
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
        count=count+1
    return y/count

# Helper visualization functions
def plot_loss_surface(training_data, x_range=(-5, 15), y_range=(-5, 5), resolution=100):
    """Plot the loss function surface and contour."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), training_data)
    
    return X, Y, Z

def visualize_loss_function():
    """
    Generate training data and visualize the loss function surface and contour.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    training_data = generate_trainingdata(25)
    
    # First, explore some function values to determine a good plotting range
    test_points = [
        (-5, -5), (0, 0), (5, 0), (10, 0), (10, 5), (0, 5), (5, 5)
    ]
    
    print("Function values at various points:")
    for x, y in test_points:
        point = np.array([x, y])
        value = f(point, training_data)
        print(f"f([{x}, {y}]) = {value:.4f}")
    
    # Based on exploration, choose appropriate ranges for visualization
    # The loss landscape features are between these ranges
    x_range = (-5, 15)
    y_range = (-5, 5)
    
    print(f"\nPlotting loss function over x range {x_range} and y range {y_range}")
    print("This range was chosen to capture the minimum and important features of the loss landscape.")
    
    # Generate loss surface
    X, Y, Z = plot_loss_surface(training_data, x_range, y_range, resolution=50)
    
    # Plot wireframe
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Function Surface (Full Training Data)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig('loss_surface_wireframe.png')
    plt.show()
    plt.close(fig)
    
    # Plot contour
    fig = plt.figure(figsize=(12, 10))
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title('Loss Function Contour (Full Training Data)')
    # plt.savefig('loss_surface_contour.png')
    plt.show()
    plt.close(fig)
    
    # Find approximate minimum
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_idx]
    min_y = Y[min_idx]
    min_z = Z[min_idx]
    print(f"\nApproximate minimum found at: [{min_x}, {min_y}] with value {min_z:.4f}")
    
    return X, Y, Z, (min_x, min_y, min_z)

# Run visualization
if __name__ == "__main__":
    X, Y, Z, min_point = visualize_loss_function()

# %%
## to test the data generation part

# %%
np.random.seed(42)  # For reproducibility
training_data = generate_trainingdata(25)
print("First few points of training data:", training_data[:3])

# %%
# to see the the local minima

# %%
# Test more points systematically
x_values = np.linspace(-10, 15, 26)
y_values = np.linspace(-10, 5, 16)
min_loss = float('inf')
min_point = None

for x in x_values:
    for y in y_values:
        point = np.array([x, y])
        loss_val = f(point, training_data)
        if loss_val < min_loss:
            min_loss = loss_val
            min_point = point

print(f"Approximate minimum found at: {min_point} with value {min_loss:.4f}")

# %%
#### a(iii) Calculate Derivatives Using Finite Differences (1 plot) gradient_field.png

# %%
import numpy as np
import matplotlib.pyplot as plt

# Functions from previous code
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
        count=count+1
    return y/count

def finite_difference_gradient(func, x, minibatch, h=1e-6):
    """
    Calculate the gradient of func at point x using finite differences.
    
    Parameters:
    - func: The function to differentiate
    - x: The point at which to calculate the gradient
    - minibatch: The mini-batch to use for the function
    - h: Step size for finite difference
    
    Returns:
    - grad: The gradient vector [∂f/∂x₁, ∂f/∂x₂]
    """
    grad = np.zeros_like(x)
    f_x = func(x, minibatch)
    
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (func(x_plus_h, minibatch) - f_x) / h
    
    return grad

def finite_difference_hessian(func, x, minibatch, h=1e-5):
    """
    Calculate the Hessian matrix of func at point x using finite differences.
    
    Parameters:
    - func: The function to differentiate
    - x: The point at which to calculate the Hessian
    - minibatch: The mini-batch to use for the function
    - h: Step size for finite difference
    
    Returns:
    - hess: The Hessian matrix [[∂²f/∂x₁², ∂²f/∂x₁∂x₂], [∂²f/∂x₂∂x₁, ∂²f/∂x₂²]]
    """
    n = len(x)
    hess = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_plus_h_i = x.copy()
            x_plus_h_i[i] += h
            
            x_plus_h_j = x.copy()
            x_plus_h_j[j] += h
            
            x_plus_h_ij = x.copy()
            x_plus_h_ij[i] += h
            x_plus_h_ij[j] += h
            
            # Mixed partial derivative
            if i == j:
                # For diagonal elements, use the standard second derivative formula
                hess[i, j] = (func(x_plus_h_i + h, minibatch) - 2 * func(x_plus_h_i, minibatch) + func(x, minibatch)) / (h * h)
            else:
                # For off-diagonal elements, use the mixed partial derivative formula
                hess[i, j] = (func(x_plus_h_ij, minibatch) - func(x_plus_h_i, minibatch) - func(x_plus_h_j, minibatch) + func(x, minibatch)) / (h * h)
    
    return hess

def validate_gradient(x_values, training_data):
    """
    Calculate and print gradients at multiple points to validate the implementation.
    
    Parameters:
    - x_values: List of points at which to calculate gradients
    - training_data: Training data to use for the function
    """
    print("Gradients at various points:")
    for x in x_values:
        grad = finite_difference_gradient(f, np.array(x), training_data)
        print(f"∇f({x}) = {grad}")

def analyze_derivatives():
    """
    Analyze the derivatives of the loss function.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    training_data = generate_trainingdata(25)
    
    # Points to analyze
    points = [
        (-2, -2),  # Far from minimum
        (0, 0),    # Near the center
        (9, 0),    # Near expected minimum based on visualization
        (9.5, 0.5) # Another point near minimum
    ]
    
    # Validate gradients
    validate_gradient(points, training_data)
    
    # Calculate and visualize gradients on the loss surface
    # We'll create a vector field showing gradient directions
    x_range = (-5, 15)
    y_range = (-5, 5)
    grid_size = 10
    
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values for contour plot
    Z = np.zeros((grid_size, grid_size))
    U = np.zeros((grid_size, grid_size))  # x-component of gradient
    V = np.zeros((grid_size, grid_size))  # y-component of gradient
    
    for i in range(grid_size):
        for j in range(grid_size):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(point, training_data)
            grad = finite_difference_gradient(f, point, training_data)
            U[i, j] = -grad[0]  # Negative because gradient points in direction of steepest ascent
            V[i, j] = -grad[1]  # But we want direction of steepest descent
    
    # Plot contour with gradient vectors
    plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.4)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    # Normalize the gradient vectors for better visualization
    magnitude = np.sqrt(U**2 + V**2)
    max_magnitude = np.max(magnitude)
    U = U / max_magnitude
    V = V / max_magnitude
    
    # Plot gradient vectors
    plt.quiver(X, Y, U, V, color='red', width=0.002, scale=30)
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title('Loss Function Contour with Gradient Directions')
    plt.show()
    # plt.savefig('gradient_field.png')
    plt.close()
    
    # Calculate Hessian at the approximate minimum to analyze curvature
    min_point = np.array([9, 0])  # Approximate minimum from visualization
    hessian = finite_difference_hessian(f, min_point, training_data)
    
    print("\nHessian matrix at approximate minimum:", min_point)
    print(hessian)
    
    # Calculate eigenvalues to analyze curvature at the minimum
    eigvals = np.linalg.eigvals(hessian)
    print("\nEigenvalues of the Hessian:", eigvals)
    print(f"Condition number: {max(abs(eigvals))/min(abs(eigvals)):.4f}")
    
    # If eigenvalues are positive, the point is a local minimum
    if np.all(eigvals > 0):
        print("All eigenvalues are positive, confirming this is a local minimum.")
    else:
        print("Not all eigenvalues are positive, this might not be a minimum.")

# Run derivative analysis
if __name__ == "__main__":
    analyze_derivatives()

# %%
### b (i) Gradient Descent with Constant Step Size (2 plots) gd_trajectory.png, gd_loss_convergence.png

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Functions from previous code
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
        count=count+1
    return y/count

def gradient_f(x, minibatch, h=1e-6):
    """Calculate gradient of f using finite differences."""
    grad = np.zeros(2)
    for i in range(2):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (f(x_plus_h, minibatch) - f(x, minibatch)) / h
    return grad

def plot_loss_surface(training_data, x_range=(-5, 15), y_range=(-5, 5), resolution=50):
    """Plot the loss function surface and contour."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), training_data)
    
    return X, Y, Z

def plot_contour(X, Y, Z, trajectory=None, title='Loss Function Contour'):
    """Plot a contour map of the loss function with optional trajectory."""
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    # Add trajectory if provided
    if trajectory is not None:
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, 'r.-', linewidth=2, markersize=10, label='Optimization Path')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start Point')
        plt.plot(traj_x[-1], traj_y[-1], 'bo', markersize=10, label='End Point')
        plt.legend()
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(title)
    return fig

def gradient_descent_constant_step(initial_x, training_data, step_size, max_iterations=100):
    """
    Standard gradient descent with constant step size.
    
    Parameters:
    - initial_x: Starting point (numpy array)
    - training_data: Full training dataset
    - step_size: Constant learning rate
    - max_iterations: Maximum number of iterations
    
    Returns:
    - x: Final position
    - losses: List of loss values at each iteration
    - trajectory: List of positions at each iteration
    """
    x = initial_x.copy()
    losses = [f(x, training_data)]
    trajectory = [x.copy()]
    
    for i in range(max_iterations):
        # Calculate gradient using the full training data
        grad = gradient_f(x, training_data)
        
        # Update x
        x = x - step_size * grad
        
        # Store loss and trajectory
        losses.append(f(x, training_data))
        trajectory.append(x.copy())
    
    return x, losses, trajectory

def run_gradient_descent_experiment():
    """
    Run gradient descent with constant step size and analyze the results.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    training_data = generate_trainingdata(25)
    
    # Initial point
    initial_x = np.array([3.0, 3.0])
    
    # Step size selection
    # We'll test a few step sizes to find an appropriate one
    step_sizes = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Store results for each step size
    results = {}
    
    for step_size in step_sizes:
        final_x, losses, trajectory = gradient_descent_constant_step(
            initial_x, training_data, step_size, max_iterations=50
        )
        results[step_size] = (final_x, losses, trajectory)
        
        print(f"Step size {step_size}:")
        print(f"  Final x: {final_x}")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
    
    # Compute loss surface for visualization
    X, Y, Z = plot_loss_surface(training_data)
    
    # Select the best step size based on results
    best_step_size = min(results.keys(), key=lambda ss: results[ss][1][-1])
    print(f"\nBest step size: {best_step_size}")
    
    # Plot convergence for the best step size
    _, best_losses, best_trajectory = results[best_step_size]
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_losses, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Iterations (Step size = {best_step_size})')
    plt.grid(True)
    plt.show()
    # plt.savefig('gd_loss_convergence.png')
    plt.close()
    
    # Plot trajectory on contour plot
    fig = plot_contour(X, Y, Z, best_trajectory, 
                     title=f'Gradient Descent Trajectory (Step size = {best_step_size})')
    # plt.savefig('gd_trajectory.png')
    plt.show()
    plt.close(fig)
    
    return best_step_size, results[best_step_size]

# Run gradient descent experiment
if __name__ == "__main__":
    best_step_size, (final_x, losses, trajectory) = run_gradient_descent_experiment()

# %%
### b (ii,iii,iv) mini-batch SGD with different batch sizes to compare with the gradient descent:
### sgd_loss_convergence.png, sgd_multiple_runs.png, batch_size_comparison.png

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Functions from previous code
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
        count=count+1
    return y/count

def gradient_f(x, minibatch, h=1e-6):
    """Calculate gradient of f using finite differences."""
    grad = np.zeros(2)
    for i in range(2):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (f(x_plus_h, minibatch) - f(x, minibatch)) / h
    return grad

def plot_loss_surface(training_data, x_range=(-5, 15), y_range=(-5, 5), resolution=50):
    """Plot the loss function surface and contour."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), training_data)
    
    return X, Y, Z

def plot_contour(X, Y, Z, trajectory=None, title='Loss Function Contour'):
    """Plot a contour map of the loss function with optional trajectory."""
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    # Add trajectory if provided
    if trajectory is not None:
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, 'r.-', linewidth=2, markersize=10, label='Optimization Path')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start Point')
        plt.plot(traj_x[-1], traj_y[-1], 'bo', markersize=10, label='End Point')
        plt.legend()
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(title)
    return fig

def sgd_constant_step(initial_x, training_data, step_size, batch_size, max_iterations=100):
    """
    Mini-batch SGD with constant step size.
    
    Parameters:
    - initial_x: Starting point (numpy array)
    - training_data: Full training dataset
    - step_size: Learning rate
    - batch_size: Size of mini-batches
    - max_iterations: Maximum number of iterations
    
    Returns:
    - x: Final position
    - losses: List of loss values at each iteration
    - trajectory: List of positions at each iteration
    """
    x = initial_x.copy()
    losses = [f(x, training_data)]  # Evaluate on full dataset for consistency
    trajectory = [x.copy()]
    
    for i in range(max_iterations):
        # Randomly select a mini-batch
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        mini_batch = training_data[indices]
        
        # Calculate gradient on mini-batch
        grad = gradient_f(x, mini_batch)
        
        # Update x
        x = x - step_size * grad
        
        # Store loss and trajectory (evaluate loss on full dataset for fair comparison)
        losses.append(f(x, training_data))
        trajectory.append(x.copy())
    
    return x, losses, trajectory

def run_sgd_experiment(best_step_size):
    """
    Run mini-batch SGD experiment with various batch sizes and analyze results.
    
    Parameters:
    - best_step_size: The best step size found from gradient descent experiment
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    training_data = generate_trainingdata(25)
    
    # Initial point
    initial_x = np.array([3.0, 3.0])
    
    # Compute loss surface for visualization
    X, Y, Z = plot_loss_surface(training_data)
    
    # Part (b)(ii): Mini-batch SGD with batch size 5
    # Run multiple times to observe variance
    batch_size = 5
    num_runs = 5
    
    print(f"\nRunning SGD with batch size {batch_size} and step size {best_step_size}")
    
    # Store results for multiple runs
    sgd_results = []
    
    plt.figure(figsize=(10, 6))
    
    for run in range(num_runs):
        final_x, losses, trajectory = sgd_constant_step(
            initial_x, training_data, best_step_size, batch_size, max_iterations=100
        )
        sgd_results.append((final_x, losses, trajectory))
        
        print(f"Run {run+1}:")
        print(f"  Final x: {final_x}")
        print(f"  Final loss: {losses[-1]:.6f}")
        
        # Plot loss convergence
        plt.plot(losses, linewidth=1, label=f'Run {run+1}')
        
        # Plot trajectory
        fig = plot_contour(X, Y, Z, trajectory,
                         title=f'SGD Trajectory - Run {run+1} (batch size={batch_size}, η={best_step_size})')
        plt.savefig(f'sgd_trajectory_run{run+1}.png')
        plt.close(fig)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'SGD Loss Convergence (batch size={batch_size}, η={best_step_size})')
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.savefig('sgd_loss_convergence.png')
    plt.close()
    
    # Plot all trajectories on the same contour
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    colors = ['r', 'g', 'b', 'm', 'c']
    for i, (_, _, trajectory) in enumerate(sgd_results):
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, f'{colors[i%len(colors)]}.-', linewidth=1, markersize=3, 
                label=f'Run {i+1}')
    
    plt.plot(initial_x[0], initial_x[1], 'ko', markersize=10, label='Start Point')
    plt.legend()
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(f'Multiple SGD Runs (batch size={batch_size}, η={best_step_size})')
    # plt.savefig('sgd_multiple_runs.png')
    plt.show()
    plt.close()
    
    # Part (b)(iii): Vary batch size
    batch_sizes = [1, 5, 10, 20, 25]  # 25 = full batch (GD)
    
    batch_results = {}
    
    for bs in batch_sizes:
        print(f"\nRunning SGD with batch size {bs}")
        final_x, losses, trajectory = sgd_constant_step(
            initial_x, training_data, best_step_size, bs, max_iterations=100
        )
        batch_results[bs] = (final_x, losses, trajectory)
        
        print(f"Batch size {bs}:")
        print(f"  Final x: {final_x}")
        print(f"  Final loss: {losses[-1]:.6f}")
        
        # Plot trajectory
        fig = plot_contour(X, Y, Z, trajectory,
                         title=f'SGD Trajectory (batch size={bs}, η={best_step_size})')
        plt.savefig(f'sgd_trajectory_bs{bs}.png')
        plt.close(fig)
    
    # Plot loss convergence for different batch sizes
    plt.figure(figsize=(10, 6))
    for bs, (_, losses, _) in batch_results.items():
        plt.plot(losses, linewidth=2, label=f'Batch Size {bs}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Iterations for Different Batch Sizes (η={best_step_size})')
    plt.grid(True)
    plt.legend()
    # plt.savefig('batch_size_comparison.png')
    plt.show()
    plt.close()
    
    # Plot all final positions on the same contour
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    for bs, (final_x, _, trajectory) in batch_results.items():
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, '.-', linewidth=1, markersize=3, 
                label=f'Batch Size {bs}')
        plt.plot(final_x[0], final_x[1], 'o', markersize=8)
    
    plt.plot(initial_x[0], initial_x[1], 'ko', markersize=10, label='Start Point')
    plt.legend()
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(f'Effect of Batch Size (η={best_step_size})')
    plt.show()
    # plt.savefig('batch_size_effect.png')
    plt.close()
    
    # Part (b)(iv): Vary step size with fixed batch size
    step_sizes = [0.01, 0.05, 0.1, 0.2, 0.5]
    fixed_batch_size = 5
    
    step_results = {}
    
    for ss in step_sizes:
        print(f"\nRunning SGD with step size {ss} and batch size {fixed_batch_size}")
        final_x, losses, trajectory = sgd_constant_step(
            initial_x, training_data, ss, fixed_batch_size, max_iterations=100
        )
        step_results[ss] = (final_x, losses, trajectory)
        
        print(f"Step size {ss}:")
        print(f"  Final x: {final_x}")
        print(f"  Final loss: {losses[-1]:.6f}")
        
        # Plot trajectory
        fig = plot_contour(X, Y, Z, trajectory,
                         title=f'SGD Trajectory (batch size={fixed_batch_size}, η={ss})')
        plt.savefig(f'sgd_trajectory_ss{ss}.png')
        plt.close(fig)
    
    # Plot loss convergence for different step sizes
    plt.figure(figsize=(10, 6))
    for ss, (_, losses, _) in step_results.items():
        plt.plot(losses, linewidth=2, label=f'Step Size {ss}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Iterations for Different Step Sizes (batch size={fixed_batch_size})')
    plt.grid(True)
    plt.legend()
    # plt.savefig('step_size_comparison.png')
    plt.show()
    plt.close()
    
    # Plot all final positions on the same contour
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    for ss, (final_x, _, trajectory) in step_results.items():
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, '.-', linewidth=1, markersize=3, 
                label=f'Step Size {ss}')
        plt.plot(final_x[0], final_x[1], 'o', markersize=8)
    
    plt.plot(initial_x[0], initial_x[1], 'ko', markersize=10, label='Start Point')
    plt.legend()
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(f'Effect of Step Size (batch size={fixed_batch_size})')
    # plt.savefig('step_size_effect.png')
    plt.show()
    plt.close()
    
    return batch_results, step_results

# Run SGD experiment
if __name__ == "__main__":
    # Assume we have the best step size from the previous experiment
    best_step_size = 0.1  # This should be the result from gradient_descent experiment
    batch_results, step_results = run_sgd_experiment(best_step_size)

# %%
#part c implement the various optimization methods (Polyak, RMSProp, Heavy Ball, and Adam) 

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Functions from previous code
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(38*(z[0]**2+z[1]**2), (z[0]+8)**2+(z[1]+3)**2)   
        count=count+1
    return y/count

def gradient_f(x, minibatch, h=1e-6):
    """Calculate gradient of f using finite differences."""
    grad = np.zeros(2)
    for i in range(2):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (f(x_plus_h, minibatch) - f(x, minibatch)) / h
    return grad

def plot_loss_surface(training_data, x_range=(-5, 15), y_range=(-5, 5), resolution=50):
    """Plot the loss function surface and contour."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), training_data)
    
    return X, Y, Z

def plot_contour(X, Y, Z, trajectory=None, title='Loss Function Contour'):
    """Plot a contour map of the loss function with optional trajectory."""
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    # Add trajectory if provided
    if trajectory is not None:
        traj_x = [point[0] for point in trajectory]
        traj_y = [point[1] for point in trajectory]
        plt.plot(traj_x, traj_y, 'r.-', linewidth=2, markersize=10, label='Optimization Path')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start Point')
        plt.plot(traj_x[-1], traj_y[-1], 'bo', markersize=10, label='End Point')
        plt.legend()
    
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(title)
    return fig

def polyak_sgd(initial_x, training_data, f_star, batch_size, max_step=0.2, max_iterations=100):
    """
    Mini-batch SGD with Polyak step size.
    
    Parameters:
    - initial_x: Starting point (numpy array)
    - training_data: Full training dataset
    - f_star: Optimal function value (approximate)
    - batch_size: Size of mini-batches
    - max_step: Maximum allowed step size
    - max_iterations: Maximum number of iterations
    
    Returns:
    - x: Final position
    - losses: List of loss values at each iteration
    - trajectory: List of positions at each iteration
    """
    x = initial_x.copy()
    losses = [f(x, training_data)]  # Evaluate on full dataset for consistency
    trajectory = [x.copy()]
    
    for i in range(max_iterations):
        # Randomly select a mini-batch
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        mini_batch = training_data[indices]
        
        # Calculate gradient on mini-batch
        grad = gradient_f(x, mini_batch)
        
        # Calculate current loss on mini-batch
        current_f = f(x, mini_batch)
        
        # Calculate Polyak step size
        grad_norm_squared = np.sum(grad**2)
        if grad_norm_squared < 1e-10 or current_f <= f_star:
            step = 0  # No step if gradient is very small or already at optimum
        else:
            step = min((current_f - f_star) / grad_norm_squared, max_step)
        
        # Update x
        x = x - step * grad
        
        # Store loss and trajectory (evaluate loss on full dataset for fair comparison)
        losses.append(f(x, training_data))
        trajectory.append(x.copy())
    
    return x, losses, trajectory

def rmsprop_sgd(initial_x, training_data, step_size, batch_size, beta=0.9, epsilon=1e-8, max_iterations=100):
    """
    Mini-batch SGD with RMSProp.
    
    Parameters:
    - initial_x: Starting point (numpy array)
    - training_data: Full training dataset
    - step_size: Base learning rate
    - batch_size: Size of mini-batches
    - beta: Decay rate for moving average
    - epsilon: Small constant to avoid division by zero
    - max_iterations: Maximum number of iterations
    
    Returns:
    - x: Final position
    - losses: List of loss values at each iteration
    - trajectory: List of positions at each iteration
    """
    x = initial_x.copy()
    losses = [f(x, training_data)]  # Evaluate on full dataset for consistency
    trajectory = [x.copy()]
    
    # Initialize squared gradient accumulator
    s = np.zeros_like(x)
    
    for i in range(max_iterations):
        # Randomly select a mini-batch
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        mini_batch = training_data[indices]
        
        # Calculate gradient on mini-batch
        grad = gradient_f(x, mini_batch)
        
        # Update squared gradient accumulator
        s = beta * s + (1 - beta) * grad**2
        
        # Calculate adaptive step size
        adjusted_step = step_size / (np.sqrt(s) + epsilon)
        
        # Update x
        x = x - adjusted_step * grad
        
        # Store loss and trajectory (evaluate loss on full dataset for fair comparison)
        losses.append(f(x, training_data))
        trajectory.append(x.copy())
    
    return x, losses, trajectory

def heavy_ball_sgd(initial_x, training_data, step_size, batch_size, beta=0.9, max_iterations=100):
    """
    Mini-batch SGD with Heavy Ball (momentum).
    
    Parameters:
    - initial_x: Starting point (numpy array)
    - training_data: Full training dataset
    - step_size: Learning rate
    - batch_size: Size of mini-batches
    - beta: Momentum parameter
    - max_iterations: Maximum number of iterations
    
    Returns:
    - x: Final position
    - losses: List of loss values at each iteration
    - trajectory: List of positions at each iteration
    """
    x = initial_x.copy()
    losses = [f(x, training_data)]  # Evaluate on full dataset for consistency
    trajectory = [x.copy()]
    
    # Initialize velocity
    v = np.zeros_like(x)
    
    for i in range(max_iterations):
        # Randomly select a mini-batch
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        mini_batch = training_data[indices]
        
        # Calculate gradient on mini-batch
        grad = gradient_f(x, mini_batch)
        
        # Update with momentum
        v = beta * v - step_size * grad
        x = x + v
        
        # Store loss and trajectory (evaluate loss on full dataset for fair comparison)
        losses.append(f(x, training_data))
        trajectory.append(x.copy())
    
    return x, losses, trajectory

def adam_sgd(initial_x, training_data, step_size, batch_size, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=100):
    """
    Mini-batch SGD with Adam optimizer.
    
    Parameters:
    - initial_x: Starting point (numpy array)
    - training_data: Full training dataset
    - step_size: Base learning rate
    - batch_size: Size of mini-batches
    - beta1: Exponential decay rate for first moment
    - beta2: Exponential decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - max_iterations: Maximum number of iterations
    
    Returns:
    - x: Final position
    - losses: List of loss values at each iteration
    - trajectory: List of positions at each iteration
    """
    x = initial_x.copy()
    losses = [f(x, training_data)]  # Evaluate on full dataset for consistency
    trajectory = [x.copy()]
    
    # Initialize moment estimates
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    t = 0  # Timestep
    
    for i in range(max_iterations):
        t += 1
        
        # Randomly select a mini-batch
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        mini_batch = training_data[indices]
        
        # Calculate gradient on mini-batch
        grad = gradient_f(x, mini_batch)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad**2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Store loss and trajectory (evaluate loss on full dataset for fair comparison)
        losses.append(f(x, training_data))
        trajectory.append(x.copy())
    
    return x, losses, trajectory

def run_advanced_optimizer_experiment():
    """
    Run and compare the advanced optimization methods: Polyak, RMSProp, Heavy Ball, and Adam.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    training_data = generate_trainingdata(25)
    
    # Initial point
    initial_x = np.array([3.0, 3.0])
    
    # Compute loss surface for visualization
    X, Y, Z = plot_loss_surface(training_data)
    
    # Find approximate minimum value for Polyak step size
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    f_star = Z[min_idx] * 0.95  # Slightly lower than observed minimum to ensure convergence
    
    # Fixed parameters
    batch_size = 5
    max_iterations = 100
    
    # Part (c)(i): Polyak step size
    print("\nRunning SGD with Polyak step size")
    
    final_x_polyak, losses_polyak, trajectory_polyak = polyak_sgd(
        initial_x, training_data, f_star, batch_size, max_step=0.2, max_iterations=max_iterations
    )
    
    print(f"Polyak step size:")
    print(f"  Final x: {final_x_polyak}")
    print(f"  Final loss: {losses_polyak[-1]:.6f}")
    
    # Plot trajectory
    fig = plot_contour(X, Y, Z, trajectory_polyak,
                     title=f'SGD with Polyak Step Size (batch size={batch_size})')
    # plt.savefig('polyak_trajectory.png')
    plt.show()
    plt.close(fig)
    
    # Part (c)(ii): RMSProp
    print("\nRunning SGD with RMSProp")
    
    # Parameters for RMSProp
    step_size_rmsprop = 0.1
    beta_rmsprop = 0.9
    
    final_x_rmsprop, losses_rmsprop, trajectory_rmsprop = rmsprop_sgd(
        initial_x, training_data, step_size_rmsprop, batch_size, 
        beta=beta_rmsprop, epsilon=1e-8, max_iterations=max_iterations
    )
    
    print(f"RMSProp (step_size={step_size_rmsprop}, beta={beta_rmsprop}):")
    print(f"  Final x: {final_x_rmsprop}")
    print(f"  Final loss: {losses_rmsprop[-1]:.6f}")
    
    # Plot trajectory
    fig = plot_contour(X, Y, Z, trajectory_rmsprop,
                     title=f'SGD with RMSProp (batch size={batch_size})')
    # plt.savefig('rmsprop_trajectory.png')
    plt.show()
    plt.close(fig)
    
    # Part (c)(iii): Heavy Ball
    print("\nRunning SGD with Heavy Ball")
    
    # Parameters for Heavy Ball
    step_size_hb = 0.1
    beta_hb = 0.9
    
    final_x_hb, losses_hb, trajectory_hb = heavy_ball_sgd(
        initial_x, training_data, step_size_hb, batch_size, 
        beta=beta_hb, max_iterations=max_iterations
    )
    
    print(f"Heavy Ball (step_size={step_size_hb}, beta={beta_hb}):")
    print(f"  Final x: {final_x_hb}")
    print(f"  Final loss: {losses_hb[-1]:.6f}")
    
    # Plot trajectory
    fig = plot_contour(X, Y, Z, trajectory_hb,
                     title=f'SGD with Heavy Ball (batch size={batch_size})')
    # plt.savefig('heavy_ball_trajectory.png')
    plt.show()
    plt.close(fig)
    
    # Part (c)(iv): Adam
    print("\nRunning SGD with Adam")
    
    # Parameters for Adam
    step_size_adam = 0.1
    beta1_adam = 0.9
    beta2_adam = 0.999
    
    final_x_adam, losses_adam, trajectory_adam = adam_sgd(
        initial_x, training_data, step_size_adam, batch_size, 
        beta1=beta1_adam, beta2=beta2_adam, epsilon=1e-8, max_iterations=max_iterations
    )
    
    print(f"Adam (step_size={step_size_adam}, beta1={beta1_adam}, beta2={beta2_adam}):")
    print(f"  Final x: {final_x_adam}")
    print(f"  Final loss: {losses_adam[-1]:.6f}")
    
    # Plot trajectory
    fig = plot_contour(X, Y, Z, trajectory_adam,
                     title=f'SGD with Adam (batch size={batch_size})')
    # plt.savefig('adam_trajectory.png')
    plt.show()
    plt.close(fig)
    
    # Compare convergence of all methods
    plt.figure(figsize=(12, 8))
    plt.plot(losses_polyak, linewidth=2, label='Polyak')
    plt.plot(losses_rmsprop, linewidth=2, label='RMSProp')
    plt.plot(losses_hb, linewidth=2, label='Heavy Ball')
    plt.plot(losses_adam, linewidth=2, label='Adam')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Convergence Comparison of Advanced Optimization Methods (batch size={batch_size})')
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.savefig('advanced_methods_comparison.png')
    plt.close()
    
    # Compare trajectories of all methods on the same contour
    fig = plt.figure(figsize=(12, 10))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 15, colors='black', alpha=0.8)
    filled_contour = plt.contourf(X, Y, Z, 100, cmap='viridis', alpha=0.6)
    plt.colorbar(filled_contour)
    
    # Plot trajectories
    plt.plot([t[0] for t in trajectory_polyak], [t[1] for t in trajectory_polyak], 'r.-', linewidth=1, markersize=3, label='Polyak')
    plt.plot([t[0] for t in trajectory_rmsprop], [t[1] for t in trajectory_rmsprop], 'g.-', linewidth=1, markersize=3, label='RMSProp')
    plt.plot([t[0] for t in trajectory_hb], [t[1] for t in trajectory_hb], 'b.-', linewidth=1, markersize=3, label='Heavy Ball')
    plt.plot([t[0] for t in trajectory_adam], [t[1] for t in trajectory_adam], 'm.-', linewidth=1, markersize=3, label='Adam')
    
    plt.plot(initial_x[0], initial_x[1], 'ko', markersize=10, label='Start Point')
    plt.legend()
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title(f'Trajectory Comparison of Advanced Optimization Methods (batch size={batch_size})')
    plt.show()
    # plt.savefig('advanced_methods_trajectories.png')
    plt.close()
    
    # Check effect of batch size on advanced methods
    # We'll focus on Adam as an example
    batch_sizes = [1, 5, 10, 20, 25]
    
    adam_batch_results = {}
    
    for bs in batch_sizes:
        print(f"\nRunning Adam with batch size {bs}")
        final_x, losses, trajectory = adam_sgd(
            initial_x, training_data, step_size_adam, bs, 
            beta1=beta1_adam, beta2=beta2_adam, epsilon=1e-8, max_iterations=max_iterations
        )
        adam_batch_results[bs] = (final_x, losses, trajectory)
        
        print(f"Batch size {bs}:")
        print(f"  Final x: {final_x}")
        print(f"  Final loss: {losses[-1]:.6f}")
    
    # Plot loss convergence for different batch sizes with Adam
    plt.figure(figsize=(10, 6))
    for bs, (_, losses, _) in adam_batch_results.items():
        plt.plot(losses, linewidth=2, label=f'Batch Size {bs}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Adam: Loss vs. Iterations for Different Batch Sizes')
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.savefig('adam_batch_size_comparison.png')
    plt.close()
    
    return {
        'polyak': (final_x_polyak, losses_polyak, trajectory_polyak),
        'rmsprop': (final_x_rmsprop, losses_rmsprop, trajectory_rmsprop),
        'heavy_ball': (final_x_hb, losses_hb, trajectory_hb),
        'adam': (final_x_adam, losses_adam, trajectory_adam),
        'adam_batch_results': adam_batch_results
    }

# Run advanced optimizer experiment
if __name__ == "__main__":
    results = run_advanced_optimizer_experiment()

# %%



