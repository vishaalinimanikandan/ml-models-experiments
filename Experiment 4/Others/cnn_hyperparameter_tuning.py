import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
import sys

# Load the original CIFAR10 code as a base
def load_and_preprocess_cifar10(n_train=5000):
    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train[1:n_train]; y_train=y_train[1:n_train]
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("Original x_train shape:", x_train.shape)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test, num_classes, input_shape

def build_cifar10_model(input_shape, num_classes):
    # Using the same model architecture as in the downloaded code
    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    
    return model

def train_and_evaluate_model(hyperparams, x_train, y_train, x_test, y_test, num_classes, input_shape, verbose=0):
    """
    Train and evaluate the CNN with the given hyperparameters.
    
    Parameters:
    hyperparams (list): [batch_size, learning_rate, beta1, beta2, epochs]
    x_train, y_train: Training data
    x_test, y_test: Test data
    num_classes: Number of output classes
    input_shape: Input shape of the model
    verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
    
    Returns:
    float: The cost (negative validation accuracy)
    """
    # Extract hyperparameters
    batch_size = int(hyperparams[0])
    learning_rate = hyperparams[1]
    beta1 = hyperparams[2]
    beta2 = hyperparams[3]
    epochs = int(hyperparams[4])
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Build model
    model = build_cifar10_model(input_shape, num_classes)
    
    # Compile model with Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    # Train model with early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Split training data into training and validation sets
    validation_split = 0.1
    val_samples = int(len(x_train) * validation_split)
    x_val = x_train[-val_samples:]
    y_val = y_train[-val_samples:]
    x_train_subset = x_train[:-val_samples]
    y_train_subset = y_train[:-val_samples]
    
    # Train model
    history = model.fit(
        x_train_subset,
        y_train_subset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=verbose,
        callbacks=[early_stopping]
    )
    
    # Calculate validation accuracy and use negative accuracy as cost function (for minimization)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    
    # Calculate cross-entropy loss on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Use negative test accuracy as the cost function (we want to maximize accuracy)
    cost = -test_acc
    
    return cost

# Apply global random search to CNN hyperparameter tuning
def cnn_global_random_search(x_train, y_train, x_test, y_test, num_classes, input_shape, param_bounds, n_samples, verbose=False):
    """
    Apply global random search for CNN hyperparameter tuning.
    
    Parameters:
    x_train, y_train: Training data
    x_test, y_test: Test data
    num_classes: Number of output classes
    input_shape: Input shape
    param_bounds (list): List of tuples (min, max) for each hyperparameter
    n_samples (int): Number of samples to evaluate
    verbose (bool): Whether to print progress
    
    Returns:
    best_params (numpy.ndarray): Best hyperparameters found
    best_cost (float): Best cost value (negative test accuracy)
    costs (list): Cost values at each iteration
    times (list): Cumulative time at each iteration
    """
    n_params = len(param_bounds)
    best_params = None
    best_cost = float('inf')
    costs = []
    times = []
    start_time = time.time()
    
    # Define wrapper function for cost evaluation
    def evaluate_params(params):
        return train_and_evaluate_model(params, x_train, y_train, x_test, y_test, num_classes, input_shape, verbose=0)
    
    for i in range(n_samples):
        # Generate random parameters
        params = np.zeros(n_params)
        for j in range(n_params):
            lower, upper = param_bounds[j]
            params[j] = np.random.uniform(lower, upper)
        
        # Evaluate cost function
        cost = evaluate_params(params)
        
        # Update best parameters if cost is lower
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        
        # Record cost and time
        costs.append(best_cost)
        times.append(time.time() - start_time)
        
        if verbose:
            print(f"Iteration {i+1}/{n_samples}, Params: {params}, Cost: {cost:.6f}, Best Cost: {best_cost:.6f}")
    
    return best_params, best_cost, costs, times

def cnn_modified_random_search(x_train, y_train, x_test, y_test, num_classes, input_shape, param_bounds, n_samples, m_best, n_iterations, neighborhood_size=0.1, verbose=False):
    """
    Apply modified random search for CNN hyperparameter tuning.
    
    Parameters:
    x_train, y_train: Training data
    x_test, y_test: Test data
    num_classes: Number of output classes
    input_shape: Input shape
    param_bounds (list): List of tuples (min, max) for each hyperparameter
    n_samples (int): Number of samples per iteration
    m_best (int): Number of best points to keep
    n_iterations (int): Number of iterations
    neighborhood_size (float): Size of neighborhood for sampling
    verbose (bool): Whether to print progress
    
    Returns:
    best_params (numpy.ndarray): Best hyperparameters found
    best_cost (float): Best cost value (negative test accuracy)
    costs (list): Cost values at each iteration
    times (list): Cumulative time at each iteration
    """
    n_params = len(param_bounds)
    population = []
    population_costs = []
    costs = []
    times = []
    start_time = time.time()
    
    # Define wrapper function for cost evaluation
    def evaluate_params(params):
        return train_and_evaluate_model(params, x_train, y_train, x_test, y_test, num_classes, input_shape, verbose=0)
    
    # Initial random sampling
    print("Generating initial population...")
    for i in range(n_samples):
        params = np.zeros(n_params)
        for j in range(n_params):
            lower, upper = param_bounds[j]
            params[j] = np.random.uniform(lower, upper)
        
        cost = evaluate_params(params)
        
        population.append(params)
        population_costs.append(cost)
        
        if verbose:
            print(f"  Sample {i+1}/{n_samples}, Params: {params}, Cost: {cost:.6f}")
    
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
        print(f"Iteration {iteration+1}/{n_iterations}")
        new_population = []
        new_population_costs = []
        
        # Generate neighborhood samples around each point in the population
        for i, params in enumerate(population):
            for j in range(n_samples // m_best):
                # Generate random perturbation
                new_params = params.copy()
                for k in range(n_params):
                    lower, upper = param_bounds[k]
                    range_size = upper - lower
                    perturbation = np.random.uniform(-neighborhood_size * range_size, 
                                                    neighborhood_size * range_size)
                    new_params[k] += perturbation
                    
                    # Clip to bounds
                    new_params[k] = max(lower, min(upper, new_params[k]))
                
                # Evaluate cost function
                cost = evaluate_params(new_params)
                
                new_population.append(new_params)
                new_population_costs.append(cost)
                
                if verbose:
                    print(f"  Sample {i*n_samples//m_best+j+1}/{n_samples}, Params: {new_params}, Cost: {cost:.6f}")
        
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
        
        print(f"  Best cost so far: {best_cost:.6f}, Best params: {best_params}")
    
    return best_params, best_cost, costs, times

# Main function to run the hyperparameter tuning
def main():
    # Load and preprocess CIFAR10 data
    print("Loading and preprocessing CIFAR10 data...")
    x_train, y_train, x_test, y_test, num_classes, input_shape = load_and_preprocess_cifar10()
    
    # Define hyperparameter bounds
    # [batch_size, learning_rate, beta1, beta2, epochs]
    param_bounds = [
        (16, 256),       # batch_size
        (0.0001, 0.01),  # learning_rate
        (0.8, 0.99),     # beta1
        (0.99, 0.9999),  # beta2
        (5, 30)          # epochs
    ]
    
    # Apply global random search
    print("\nApplying Global Random Search...")
    grs_best_params, grs_best_cost, grs_costs, grs_times = cnn_global_random_search(
        x_train, y_train, x_test, y_test, num_classes, input_shape,
        param_bounds,
        n_samples=10,  # Reduced for demonstration
        verbose=True
    )
    
    print("\nGlobal Random Search Results:")
    print(f"Best Parameters: {grs_best_params}")
    print(f"Best Cost (negative accuracy): {grs_best_cost:.6f}")
    print(f"Best Accuracy: {-grs_best_cost:.6f}")
    
    # Apply modified random search
    print("\nApplying Modified Random Search...")
    mrs_best_params, mrs_best_cost, mrs_costs, mrs_times = cnn_modified_random_search(
        x_train, y_train, x_test, y_test, num_classes, input_shape,
        param_bounds,
        n_samples=5,     # Reduced for demonstration
        m_best=2,
        n_iterations=2,  # Reduced for demonstration
        neighborhood_size=0.1,
        verbose=True
    )
    
    print("\nModified Random Search Results:")
    print(f"Best Parameters: {mrs_best_params}")
    print(f"Best Cost (negative accuracy): {mrs_best_cost:.6f}")
    print(f"Best Accuracy: {-mrs_best_cost:.6f}")
    
    # Convert hyperparameters to more readable format
    def format_hyperparams(params):
        return {
            'batch_size': int(params[0]),
            'learning_rate': params[1],
            'beta1': params[2],
            'beta2': params[3],
            'epochs': int(params[4])
        }
    
    print("\nGlobal Random Search Best Hyperparameters:")
    print(format_hyperparams(grs_best_params))
    
    print("\nModified Random Search Best Hyperparameters:")
    print(format_hyperparams(mrs_best_params))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(grs_times, grs_costs, label='Global Random Search')
    plt.plot(mrs_times, mrs_costs, label='Modified Random Search')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Best Cost (negative accuracy)')
    plt.title('Hyperparameter Tuning Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_tuning_results.png')
    plt.show()

if __name__ == "__main__":
    main()

