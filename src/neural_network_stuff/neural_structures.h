#ifndef NEURAL_STRUCTURES
#define NEURAL_STRUCTURES

#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

void set_parameters(void); // set the weights biases and outputs for each layer.

// ReLU = max(0, x)
double relu(double value);

// ReLU' = (x <= 0) -> 0 else 1
double d_relu(double value);

// softmax(v) = exp(v) / sum(exp(v_i)) for i in v.
vector *softmax(vector *v);

// Computes the loss of a vector.
double loss_function(vector *predictions, vector *real);

// change needed in the weight (gradient).
matrix *get_weight_gradient(vector *L, vector *A);

// W := Weights of next layer, L := lambda of next layer, Z := values of the layer.
vector *get_hidden_lambda(matrix *W, vector *L, vector *Z);

// lambda of the output layer (A - Y)
vector *get_output_lambda(vector *A, vector *Y);

// apply d_relu to a vector.
vector *v_d_relu(vector *v);

// Takes a vector *v and excecutes the relu function on each value in the vector and returns a new one.
vector *v_relu(vector *v);

#endif