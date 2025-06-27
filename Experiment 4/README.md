
# 🔍 Random Search Algorithms for Optimization

This folder presents an in-depth comparison of **Global Random Search**, **Modified Random Search with Population-Based Sampling**, and **Gradient Descent**. These optimization methods are applied to mathematical test functions and extended to **CNN hyperparameter tuning on CIFAR-10**.

---

## 📦 Project Files

- `random_search_algorithms.py`: Core implementations of optimization algorithms and test functions.
- `main_experiment.py`: Runs optimization on mathematical functions and generates plots.
- `cnn_hyperparameter_tuning.py`: Applies the search algorithms to CNN training.
- `hyperparameter_tuning_results.png`: Visual chart comparing performance during CNN tuning.

---

## 📌 Highlights

- ✅ Implemented global and modified random search strategies
- ✅ Analyzed two key functions (smooth & non-differentiable)
- ✅ Compared with gradient descent analytically and visually
- ✅ Applied to CNN tuning with reproducible results

---

## 🧪 Optimization Test Functions

- **Function 1:** `f(x, y) = 6(x - 1)^4 + 8(y - 2)^2`  
  - Smooth & differentiable; ideal for gradient descent.

- **Function 2:** `f(x, y) = max(x - 1, 0) + 8|y - 2|`  
  - Non-differentiable; better suited to random search.

---

## 📊 Example Chart: CNN Hyperparameter Tuning

![Hyperparameter Tuning Results](hyperparameter_tuning_results.png)

This chart shows how Global and Modified Random Search compare during the CNN tuning process.  
**Y-axis** = Negative Accuracy (lower is better), **X-axis** = Time (s)

---

## 🤖 CNN Hyperparameters Tuned

- Batch size
- Learning rate
- Adam optimizer’s `β₁`, `β₂`
- Number of training epochs

---

## ⚙️ Requirements

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## ▶️ How to Run

### Test Function Experiments
```bash
python main_experiment.py
```

### CNN Hyperparameter Tuning
```bash
python cnn_hyperparameter_tuning.py
```

---

## 📚 Algorithms Implemented

- **Global Random Search:** Uniform random sampling
- **Modified Random Search:** Population-based adaptive sampling
- **Gradient Descent:** Derivative-based optimization for smooth landscapes

