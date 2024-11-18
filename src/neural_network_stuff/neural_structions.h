#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

typedef struct {
    double value; // Activation value.
    double bias; // Could be 0.0 if the Neuron is in the input layer.
    double size_w;
    double *weights; // List of the weights, size of the next layer.
} Neuron;

typedef struct {
    Index size; // Amount of neurons in the layer.
    Neuron *neurons; // Pointer to an array of Neurons.
} Layer;

// sigmoid = 1/(1 + e^{-x})
double sigmoid(double value);

// Takes *l which is a pointer to the layer you're on and *N which is a pointer to the next Neuron.
double next_neuron_value(Layer *l, Layer* n_l, int index);

// Takes a vector *v and excecutes the sigmoid function on each value in the vector and returns a new one.
vector* v_sigmoid(vector *v);