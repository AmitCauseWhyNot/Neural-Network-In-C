#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

/*
    Va1 = sigmoid(W * Va0 + b);

    Va1: the resulting vector with size n (output activations).
    W: an n x m weight matrix, where n is the size of the output layer (matches Va1) and m is the size of the input layer (matches Va0).
    Va0: the input vector from the previous layer, with size m.
    b: the bias vector that has the biases of the next layer Va1, with size n.
*/

typedef struct {
    double value; // Activation value.
    double bias; // Could be 0.0 if the Neuron is in the input layer.
    double *weights; // List of the weights, size of the next layer.
} Neuron;

typedef struct {
    Index size;
    Neuron *neurons;
} Layer;

// sigmoid = 1/(1 + e^{-x})
double sigmoid(double value);

// Takes *l which is a pointer to the layer you're on and *N which is a pointer to the next Neuron.
double next_neuron_value(Layer *l, Neuron* N);

// Takes a vector *v and excecutes the sigmoid function on each value in the vector and returns a new one.
vector* v_sigmoid(vector *v);