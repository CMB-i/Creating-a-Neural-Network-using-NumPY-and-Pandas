# NumPy Neural Network from Scratch — Breast Cancer & MNIST

This was the task assigned:

Create a Neural Network to solve a classification problem with only using NumPY and Pandas.
**Success Criterion :**
1) Build the Core Network: Start by creating a simple, single-script neural network using only Python with NumPy and Pandas. The goal is your network should be able to successfully perform classification tasks on a simple dataset.

2) Create a Reusable Library: Refactor the initial script into an object-oriented, Pytorch-like library. The key is to create classes like Sequential to manage the model and DenseLayer for network layers, along with methods like .add(), .compile(), and .fit().

3) Demonstrate with a Real-World Problem: Use your new library to build and train a model for a practical use case. A great choice is classifying handwritten digits from the MNIST dataset to prove the library's effectiveness and scalability.

4) Create a Project Showcase Video: Produce a short (<5 min) video explaining the project. The video should outline the necessary learning path, demonstrate the library in action with the MNIST example, and discuss potential future enhancements like adding new optimizers or layer types.

   ----

This repository contains a **pure NumPy implementation** of a small neural network framework and end-to-end demos on:

- **Breast Cancer Wisconsin Dataset** (binary classification)
- **MNIST Handwritten Digit Dataset** (multi-class classification, softmax output)
- **Custom Learning Rate Decay Scheduler**
- Training/Validation/Test splits without data leakage
- Optional **visualizations**: loss curves, confusion matrix, per-class metrics

## Features

- **From-scratch neural network library** (no TensorFlow/PyTorch)
- Modular `Sequential` + `Dense` layers
- Activation functions: ReLU, Sigmoid, Softmax
- Loss functions: Binary Cross-Entropy, Categorical Cross-Entropy
- Optimizers: SGD, SGD with Weight Decay & Learning Rate Decay
- Early stopping support
- Data preprocessing helpers (standardization, one-hot encoding, flattening)
- Works entirely in **Google Colab** or any Python 3 environment

---


## Getting Started

### Clone the repo

 ⁠bash
git clone https://github.com/yourusername/numpy-nn-breast-mnist.git
cd numpy-nn-breast-mnist


⁠ ### install dependencies

This project only uses:

 ⁠bash
pip install numpy pandas scikit-learn matplotlib tensorflow


> **Note:** TensorFlow is only used for loading MNIST data — the model itself is pure NumPy.

### Run in Google Colab

* Upload `notebook.ipynb` to Colab
* Run cells **in order** (Cells 1 → 6)
* For MNIST: The merged training cell automatically handles validation split and LR decay.

---

## Datasets

### Breast Cancer (binary classification)

* 30 features (real-valued)
* Labels: Malignant (0) / Benign (1)
* Standardized using **train split mean/std** (no leakage)

### MNIST (multi-class)

* 28×28 grayscale images
* Flattened to vectors of length 784
* Pixel values scaled to **\[0,1]**

---

## Example Results

**Breast Cancer**


Test Accuracy: ~97%


**MNIST**


Test Accuracy: ~96% (with LR decay)


---

## Visualizations

This repo can generate:

* **Loss curves** (train vs. validation)
* **Learning rate schedule**
* **Confusion matrix** (counts & normalized)
* **Per-class metrics** (precision, recall, F1)
* **Misclassified examples** preview

Example (MNIST misclassified):

<img width="758" height="788" alt="image" src="https://github.com/user-attachments/assets/613fadd2-7bb1-42af-9561-e2fc6a28dbe9" />


---

## How It Works

* **Forward pass**: Compute layer activations
* **Loss computation**: Compare predictions with ground truth
* **Backward pass**: Compute gradients via chain rule
* **Optimizer step**: Update parameters (optionally with weight decay / LR schedule)

---


## Future Improvements

* Add convolutional layers
* Implement Adam optimizer
* Add dropout & batch normalization
* Support custom dataset loading

---
## DEMO VIDEO:

https://youtu.be/cTb9URz_haQ

```
