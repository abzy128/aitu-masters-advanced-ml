# Backpropagation and Gradient Descent

**Submission format:** .ipynb Jupyter Notebook

## Objective

The goal of this laboratory work is to understand how neural networks learn through gradient descent and backpropagation. Students will:

- Implement gradient descent from scratch
- Visualize loss surfaces and gradients
- Observe the effect of learning rate
- Train a simple neural network and analyze weight updates

This lab is 80% hands-on coding and 20% theory-focused questions to highlight learning outcomes.

---

## Part 1 – Gradient Descent Fundamentals (From Scratch)

### 1.1 Simple Linear Regression

We start with a simple dataset and implement gradient descent manually.

**Dataset:**  
\( y = 3x + 2 + \text{noise} \)

**Task:**
- Generate a synthetic dataset
- Define the Mean Squared Error (MSE) loss
- Implement gradient descent to learn parameters \( w \) and \( b \)

**Coding Tasks:**
1. Generate 100 data points
2. Initialize parameters \( w \) and \( b \) randomly
3. Compute predictions and loss
4. Compute gradients manually
5. Update parameters using gradient descent

**Hint:**
- MSE loss: `mean((y_true - y_pred)**2)`
- Gradient w.r.t. \( w \): `-2 * mean(x * (y - y_pred))`
- Gradient w.r.t. \( b \): `-2 * mean(y - y_pred)`

### 1.2 Loss Curve Visualization

**Task:**
- Track the loss value for each iteration
- Plot loss vs iterations

**Hint:**
- Store loss values in a list during training
- Use `matplotlib.pyplot.plot()`

### 1.3 Effect of Learning Rate

**Task:**
- Train the model using three learning rates:
  - 0.001
  - 0.01
  - 0.1
- Plot all loss curves on the same graph

**Hint:**
- Too small learning rate → slow convergence
- Too large learning rate → unstable training

---

## Part 2 – Visualizing Gradient Descent

### 2.1 Loss Surface Visualization (Optional but Recommended)

**Task:**
- Create a grid of values for \( w \) and \( b \)
- Compute the loss for each pair
- Plot a contour plot of the loss surface

**Hint:**
- Use `numpy.meshgrid`
- Use `plt.contour()` or `plt.contourf()`

---

## Part 3 – Backpropagation with a Neural Network

### 3.1 Dataset Preparation

**Task:**
- Load the MNIST dataset
- Normalize pixel values to [0, 1]
- Flatten input images

**Hint:**
- Use `tensorflow.keras.datasets.mnist.load_data()`
- Divide pixel values by 255.0

### 3.2 Training with Gradient Descent

**Model Architecture:**
- Input: Flattened 28×28 image
- Dense layer: 64 neurons + ReLU
- Output layer: 10 neurons + Softmax

**Task:**
- Train the model using SGD optimizer (not Adam)
- Track training loss and accuracy

**Hint:**
- Use `tf.keras.optimizers.SGD(learning_rate=0.01)`
- Compile with `SparseCategoricalCrossentropy`

### 3.3 Observing Backpropagation

**Task:**
- Use TensorFlow's automatic differentiation
- Inspect gradients of the loss w.r.t. weights

**Hint:**
- Use `tf.GradientTape()`
- Call `tape.gradient(loss, model.trainable_variables)`

---

## Part 4 – Comparing Optimizers

### 4.1 SGD vs Adam

**Task:**
- Train two identical models:
  - One with SGD
  - One with Adam
- Use the same number of epochs

**Hint:**
- Keep architecture and batch size fixed
- Only change the optimizer

### 4.2 Visualization

**Task:**
- Plot training loss and accuracy for both optimizers

---

## Part 5 – Theoretical Questions (Answer Briefly)

**Q1.** What is the role of the learning rate in gradient descent?

**Q2.** Why do we need backpropagation in multi-layer neural networks?

**Q3.** What happens if gradients become very small during training?

**Q4.** Why does Adam often converge faster than plain SGD?

---

## Expected Learning Outcomes

By the end of this lab, students should be able to:

- Explain gradient descent mathematically and intuitively
- Implement gradient descent from scratch
- Understand how backpropagation updates neural network weights
- Analyze the effect of learning rate and optimizer choice

---

## What to Submit

A .ipynb notebook containing:

- All code implementations
- Plots and visualizations
- Short written answers to theoretical questions
```

