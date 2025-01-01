import random

# Example Dataset
data = [(x, 6*x + 9) for x in range(100)]  # y = 2x + 1

# Normalize the dataset
max_x = max(x for x, _ in data)
max_y = max(y for _, y in data)
data = [(x / max_x, y / max_y) for x, y in data]  # Scale to [0, 1]

# Parameters
learning_rate = 0.01  # Reduced learning rate
epochs = 1147
batch_size = 10  # Mini-batch size
m, b = 0, 0  # Initialize weights

# Loss function and gradient computation
def compute_loss(batch, m, b):
    return sum((y - (m*x + b))**2 for x, y in batch) / len(batch)

def compute_gradients(batch, m, b):
    m_grad = -2 * sum(x * (y - (m*x + b)) for x, y in batch) / len(batch)
    b_grad = -2 * sum((y - (m*x + b)) for x, y in batch) / len(batch)
    return m_grad, b_grad

# Mini-Batch Gradient Descent
for epoch in range(epochs):
    random.shuffle(data)  # Shuffle data
    mini_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    for batch in mini_batches:
        m_grad, b_grad = compute_gradients(batch, m, b)
        m -= learning_rate * m_grad
        b -= learning_rate * b_grad

    # Compute loss for reporting
    loss = compute_loss(data, m, b)

# Rescale parameters to original scale
m_rescaled = m * max_y / max_x
b_rescaled = b * max_y
print(f"Final parameters: m = {m_rescaled:.4f}, b = {b_rescaled:.4f}")
