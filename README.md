# Neural Network in C

[![C](https://img.shields.io/badge/language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language)) [![Visual Studio Code](https://img.shields.io/badge/IDE-VSCode-blue.svg)](https://code.visualstudio.com/) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://shields.io)

## Description

This project implements a simple feedforward neural network written in C. The network leverages matrix and vector operations to handle forward propagation, backpropagation, and gradient descent for training. It also includes functions for handling the MNIST dataset and various utility functions for neural network computations.

### Features
- Matrix and vector algebra operations
- Neural network forward propagation
- Loss computation
- Gradient computation and backpropagation
- Parameter saving/loading
- MNIST dataset support

---

## File Structure

```
src/
│
├── linear_algebra_stuff/
│   ├── matrix_stuff/
│   │   ├── matrix.h
│   │   └── matrix.c
│   
│   ├── vector_stuff/
│       ├── vector.h
│       └── vector.c
│
├── neural_network_stuff/
│   ├── neural_structures.h
│   └── neural_structures.c
│
├── MNIST_stuff/
│   ├── mnist.h
│   └── mnist.c
│
└── main.c
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amitcausewhynot/Basic-Neural-Network-In-C.git
   cd Basic-Neural-Network-In-C
   ```

2. Compile the project:
   ```bash
   gcc -o neural_network main.c linear_algebra_stuff/matrix_stuff/matrix.c \
       linear_algebra_stuff/vector_stuff/vector.c neural_network_stuff/neural_structures.c \
       MNIST_stuff/mnist.c -lm
   ```

3. Run the executable:
   ```bash
   ./neural_network
   ```

---

## Usage

### Training the Neural Network
1. Add the MNIST dataset files in the `src/data_stuff/` directory:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`

2. Run the program to train the neural network:
   ```bash
   ./neural_network
   ```

### Customization
- **Weights Initialization:** Modify `set_parameters()` in `neural_structures.c`.
- **Learning Rate:** Adjust in the `train()` function in `main.c`.
- **Loss Function:** Modify the `loss_function()` in `neural_structures.c`.

---

## Requirements

- GCC or any C compiler
- Visual Studio Code (recommended for development)
- MNIST dataset files

---

## API Reference

### Matrix Operations (`matrix.h`)
- `matrix *m_create(Index rows, Index cols, double **data);`  
  Creates a matrix with specified rows, columns, and optional data.

- `matrix *m_transpose(matrix *m);`  
  Returns the transpose of the given matrix.

- `double m_dot(double *row, double *col, int size);`  
  Computes the dot product of two arrays.

### Vector Operations (`vector.h`)
- `vector *v_create(Index length, double *values);`  
  Creates a vector with specified length and values.

- `vector *v_add(vector *v1, vector *v2);`  
  Returns the sum of two vectors.

- `vector *m_v_mult(matrix *m, vector *v);`  
  Multiplies a matrix by a vector and returns the resulting vector.

### Neural Network (`neural_structures.h`)
- `void set_parameters(void);`  
  Initializes weights, biases, and outputs for each layer.

- `vector *softmax(vector *v);`  
  Applies the softmax function to a vector.

- `double loss_function(vector *predictions, vector *real);`  
  Computes the loss between predicted and actual outputs.

### MNIST Handling (`mnist.h`)
- `double *get_image(char *path, int num);`  
  Extracts an image from the MNIST dataset.

- `int get_label(char *path, int num);`  
  Extracts a label from the MNIST dataset.

---

## Contribution

Contributions are welcome!  
Feel free to fork this repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
