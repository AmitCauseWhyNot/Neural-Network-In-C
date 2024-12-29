#ifndef NEURAL_STRUCTURES
#define NEURAL_STRUCTURES

#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

void set_parameters(void); // set the weights biases and outputs for each layer.

// sigmoid = 1/(1 + e^{-x})
double sigmoid(double value);

// sigmoid' = sigmoid(x) * (1 - sigmoid(x))
double d_sigmoid(double value);

// softmax(v) = exp(v) / sum(exp(v_i)) for i in v.
vector *softmax(vector *v);

// Computes the loss of a vector.
double loss_function(vector *predictions, double *real);

// change needed in the weight (gradient).
matrix *get_weight_gradient(vector *L, vector *A);

// W := Weights of next layer, L := lambda of next layer, a_f := pointer to an activation function, Z := values of the layer.
vector *get_hidden_lambda(matrix *W, vector *L, double (*a_f)(double), vector *Z);

// lambda of the output layer (A - Y)
vector *get_output_lambda(vector *A, vector *Y);

// apply d_sigmoid to a vector.
vector *v_d_sigmoid(vector *v);

// Takes a vector *v and excecutes the sigmoid function on each value in the vector and returns a new one.
vector *v_sigmoid(vector *v);

#endif