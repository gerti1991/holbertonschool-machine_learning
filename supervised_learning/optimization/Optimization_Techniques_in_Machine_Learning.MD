# Optimization Techniques in Machine Learning: A Simple Guide

![Optimization Techniques](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*WC47SQOjdxnAe9uJLnf0rw.jpeg)

*By Gertian Skerja*

*Published on April 23, 2025*

When training machine learning models, using the right optimization techniques can drastically improve performance. In this notebook, we’ll cover some of the most common optimization methods, explaining their mechanics, pros, and cons.

We’ll go through:

- **Feature Scaling**
- **Batch Normalization**
- **Mini-batch Gradient Descent**
- **Gradient Descent with Momentum**
- **RMSProp Optimization**
- **Adam Optimization**
- **Learning Rate Decay**

Let’s dive in!

---

## 1. Feature Scaling

**Mechanics**: Feature scaling, or normalization, ensures that all features contribute equally to the learning process. It transforms data to a common range (e.g., between 0 and 1) or standardizes it to have a mean of 0 and a standard deviation of 1.

**Example from the Project**: Here’s an implementation to calculate the normalization constants of a dataset:

```python
def normalization_constant(X):
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
```

Next, we use these constants to normalize the dataset:

```python
def normalize(X, m, s):
    return (X - m) / s
```

**Pros**:
- Prevents features with large ranges from dominating.
- Speeds up convergence in algorithms like gradient descent.

**Cons**:
- Needs to be applied consistently across training and testing data.

![Feature Scaling Diagram](feature-scaling-diagram.png)

*Note*: Replace `feature-scaling-diagram.png` with an actual diagram illustrating feature scaling.

---

## 2. Batch Normalization

**Mechanics**: Batch normalization normalizes the output of a neural network layer by adjusting and scaling it with two parameters, gamma and beta. It helps stabilize and speed up training by reducing internal covariate shifts.

**Example from the Project**: This is how you can create a batch normalization layer:

```python
import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    init = tf.keras.initializers.VarianceScaling(mode
```

**Pros**:
- Speeds up training.
- Reduces sensitivity to initialization.
- Acts as a regularizer.

**Cons**:
- Adds computational overhead.
- May not always benefit simple models.

![Batch Normalization Diagram](batch-normalization-diagram.png)

*Note*: Replace `batch-normalization-diagram.png` with an actual diagram illustrating batch normalization.

---

## 3. Mini-batch Gradient Descent

**Mechanics**: Mini-batch gradient descent breaks the dataset into small batches, computes gradients for each batch, and updates the model parameters accordingly. It combines the benefits of both stochastic and batch gradient descent.

**Example from the Project**: Here’s a function to create mini-batches:

```python
def create_mini_batches(X, Y, batch_size):
    m = X.shape[0]
    mini_batches = []
    X, Y = shuffle_data(X, Y)
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
```

**Pros**:
- Faster convergence compared to full-batch gradient descent.
- Reduces noise and instability in updates.

**Cons**:
- Requires careful tuning of batch size.
- Updates are noisier than full-batch gradient descent.

![Mini-batch Gradient Descent Diagram](mini-batch-gd-diagram.png)

*Note*: Replace `mini-batch-gd-diagram.png` with an actual diagram illustrating mini-batch gradient descent.

---

## 4. Gradient Descent with Momentum

**Mechanics**: This technique adds a “momentum” term to accelerate gradient descent, pushing it in directions that consistently reduce the loss. Momentum helps avoid oscillations in directions where the gradient frequently changes.

**Example from the Project**: Here’s the implementation for gradient descent with momentum:

```python
def update_variable_momentum(alpha, beta1, var, grad, v):
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
```

**Pros**:
- Accelerates convergence.
- Helps escape local minima and avoid oscillations.

**Cons**:
- Requires tuning the momentum hyperparameter.
- Can overshoot the minimum if not tuned properly.

![Momentum Diagram](momentum-diagram.png)

*Note*: Replace `momentum-diagram.png` with an actual diagram illustrating momentum in gradient descent.

---

## 5. RMSProp Optimization

**Mechanics**: RMSProp (Root Mean Square Propagation) adapts the learning rate for each parameter by dividing the learning rate by the average of recent magnitudes of the gradients for that parameter.

**Example from the Project**: Below is an example of the RMSProp update rule:

```python
def update_variable_RMSProp(alpha, beta2, epsilon, var, grad, s):
    s = beta2 * s + (1 - beta2) * grad**2
    var -= alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
```

**Pros**:
- Handles non-stationary objectives well.
- Prevents large oscillations in parameter updates.

**Cons**:
- Requires tuning additional hyperparameters.
- May not work well on very sparse gradients.

![RMSProp Diagram](rmsprop-diagram.png)

*Note*: Replace `rmsprop-diagram.png` with an actual diagram illustrating RMSProp.

---

## 6. Adam Optimization

**Mechanics**: Adam is an adaptive learning rate optimization algorithm that combines the advantages of both RMSProp and momentum. It computes two running averages for the gradient (first and second moments), and adjusts the learning rate accordingly.

**Example from the Project**: Here’s the Adam update function:

```python
def update_variable_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_corrected = v / (1 - beta1**t)
    s_corrected = s / (1 - beta2**t)
    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
```

**Pros**:
- Adaptive learning rate makes it robust for a wide range of tasks.
- Combines the advantages of momentum and RMSProp.

**Cons**:
- Requires more memory for moment estimations.
- Can sometimes generalize poorly.

![Adam Diagram](adam-diagram.png)

*Note*: Replace `adam-diagram.png` with an actual diagram illustrating Adam optimization.

---

## 7. Learning Rate Decay

**Mechanics**: Learning rate decay reduces the learning rate as training progresses, allowing for larger updates early on and smaller updates later. This ensures the model makes fine adjustments during later stages of training.

**Example from the Project**: Here’s how inverse time decay is implemented:

```python
def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
```

In TensorFlow, you can apply it using a schedule:

```python
alpha_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(alpha, decay_steps=1000, decay_rate=0.5)
```

**Pros**:
- Improves convergence by reducing the learning rate over time.
- Prevents overshooting the minimum.

**Cons**:
- Requires careful tuning of the decay schedule.
- Too aggressive decay can cause the model to get stuck.

![Learning Rate Decay Diagram](lr-decay-diagram.png)

*Note*: Replace `lr-decay-diagram.png` with an actual diagram illustrating learning rate decay.

---

## Conclusion

Each optimization technique plays a critical role in improving machine learning model performance. By using feature scaling, batch normalization, adaptive optimizers like Adam and RMSProp, and learning rate schedules, you can make your models more efficient and effective.

Implementing these methods in your projects can significantly enhance your results. Try them out and see how your model performance improves!

*Happy optimizing!*
