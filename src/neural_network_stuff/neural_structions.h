#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

/*
    Va1 = sigmoid(W * Va0 + b);

    Va1: the resulting vector with size n (output activations).
    W: an n x m weight matrix, where n is the size of the output layer (matches Va1) and m is the size of the input layer (matches Va0).
    Va0: the input vector from the previous layer, with size m.
    b: the bias vector that has the biases of the next layer Va1, with size n.
*/

typedef unsigned int Index;

typedef struct {
    double value; // Activation value.
    double bias; // Could be 0.0 if the Neuron is in the input layer.
} Neuron;

typedef struct {
    Neuron *from;
    Neuron *to;
    double value;
} Weight;

typedef struct {
    Neuron *neurons;

} Layer;

// sigmoid = 1/(1 + e^-x)
double sigmoid(double value);

vector* v_sigmoid(vector *v);