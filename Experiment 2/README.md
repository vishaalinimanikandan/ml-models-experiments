
# 🧮 Optimization Algorithm Comparison in Python

This project explores and compares four optimization algorithms — Polyak Step, RMSProp, Heavy Ball, and Adam — applied to various types of functions, including smooth and non-smooth surfaces. It includes analytical derivations, practical implementations, and visual convergence comparisons using Python.

---

## 📌 Objectives

- Analyze how different optimization methods perform on:
  - Smooth function: `f1(x, y) = 6(x - 1)^4 + 8(y - 2)^2`
  - Non-smooth function: `f2(x, y) = max(0, x - 1) + 8|y - 2|`
- Study convergence behavior using various learning rates and parameters.
- Evaluate performance on piecewise functions like ReLU.
- Use contour plots and update history to visualize optimizer trajectories.

---

## 🚀 Algorithms Implemented

- **Polyak Step Size**
- **RMSProp**
- **Heavy Ball (Momentum)**
- **Adam Optimizer**

Each method was tested for:
- Gradient effectiveness
- Convergence speed
- Behavior on smooth vs non-smooth functions
- Sensitivity to hyperparameters like α, β, β₁, and β₂

---

## 🧪 Experiments & Visuals

- ✅ Contour plots of each optimizer on `f1` and `f2`
- ✅ Comparison of update steps, iterations, and learning curves
- ✅ Behavior of optimizers on ReLU for `x₀ = -1, 1, 100`

---

## 📂 Project Structure

```
.
├── polyak.py
├── rmsprop.py
├── heavy_ball.py
├── adam.py
├── relu_analysis.py
├── README.md  ← you're here
```

---

## 📊 Key Insights

- Adam showed the most **stable and consistent convergence** across all function types.
- Heavy Ball had fast descent but suffered from **oscillations** with high momentum.
- RMSProp performed best with **high β and lower α** values.
- ReLU analysis demonstrated how zero gradients stop updates in RMSProp and Heavy Ball, while Adam attempts correction.

---

## 📬 Author

**Vishaalini Ramasamy Manikandan**  
MSc Computer Science (Data Science), Trinity College Dublin  
📧 [vishaalini70@gmail.com](mailto:vishaalini70@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/vishaalini-manikandan/)

---

📢 *Open to research, internship, and collaborative opportunities.*
