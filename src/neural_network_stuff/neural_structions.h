#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

typedef struct
{
    double value; // Activation value.
    double bias;  // Could be 0.0 if the Neuron is in the input layer.
    double size_w;
    double *weights; // List of the weights, size of the next layer.
} Neuron;

typedef struct
{
    Index size;      // Amount of neurons in the layer.
    Neuron *neurons; // Pointer to an array of Neurons.
} Layer;

// sigmoid = 1/(1 + e^{-x})
double sigmoid(double value);

// sigmoid' = sigmoid(x) * (1 - sigmoid(x))
double d_sigmoid(double value);

// softmax(v) = exp(v) / sum(exp(v_i)) for i in v.
double softmax(vector *v);

// Takes *l which is a pointer to the layer you're on and *N which is a pointer to the next Neuron.
double next_neuron_value(Layer *l, Layer *n_l, int index);

// Computes the loss of a vector.
double loss_function(vector *predictions, double *real);

// change needed in the weight (gradient).
matrix *get_weight_gradient(vector *L, vector *A);

// just returns the L, here for convenience.
vector *get_bias_gradient(vector *L);

// W := Weights of next layer, L := lambda of next layer, a_f := pointer to an activation function, Z := values of the layer.
vector *get_hidden_lambda(matrix *W, vector *L, (double)(*a_f)(double), vector *Z);

// lambda of the output layer (A - Y)
vector *get_output_lambda(vector *A, vector *Y);

// apply d_sigmoid to a vector.
vector *v_d_sigmoid(vector *v);

// Takes a vector *v and excecutes the sigmoid function on each value in the vector and returns a new one.
vector *v_sigmoid(vector *v);