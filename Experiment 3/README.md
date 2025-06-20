# 🔍 Gradient-Based Optimization Experiment

This repository contains the Week 6 assignment for CS7CS2: exploring **gradient-based optimization methods** using a non-convex loss surface. The loss function presents multiple local minima, which allows us to compare the behavior of several optimization strategies.

---

## 📌 Objective

- Implement and analyze optimization algorithms including:
  - Gradient Descent (GD) with constant step size
  - Mini-batch Stochastic Gradient Descent (SGD)
  - Advanced optimizers: Polyak, RMSProp, Heavy Ball, Adam
- Explore the effects of step size and batch size
- Visualize loss surfaces, convergence paths, and gradient fields

---

## 🧪 Visual Summary

![optimizer_convergence_plot](https://github.com/user-attachments/assets/5fa396f7-47c7-4727-bb0e-194d0e4b3ad3)

This graph compares the convergence speed and final loss of different optimization algorithms.

---

## 📊 Key Observations

- **Heavy Ball** consistently finds the global minimum despite oscillations.
- **Adam** and **RMSProp** quickly converge but often settle in local minima.
- **Polyak** adapts dynamically but depends on good `f*` estimation.
- **Step size** plays a more critical role than **batch size** in finding the global minimum.

---

## 🧰 Methods Implemented

- Gradient estimation using **finite differences**
- Hessian computation and curvature analysis
- Visualization of:
  - Loss surfaces (3D + contours)
  - Gradient vector fields
  - Convergence trajectories
- Comparison of multiple optimizers on the same landscape

---

