#ifndef NEURAL_STRUCTURES
#define NEURAL_STRUCTURES

#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

#define OUT_LENGTH 10
#define PI 3.141592653
#define BASE_BIAS 1e-2
#define LEAKY 0.1
#define max(x, y) (((x) >= (y)) ? (x) : (y))

typedef struct _nt {
    double value;
    double bias;
} Neuron_t;

typedef struct _lt {
    Index len;
    Neuron_t** neurons;
    matrix* weights;
} Layer_t;

void n_free(Neuron_t* n);

void l_free(Layer_t* l);

vector* get_values_vector(Layer_t* l);

vector* get_label_vector(Index lbl);

void update_bias(Layer_t* l, vector* offset, double rate);

void update_weights(Layer_t* l, matrix* offset, double rate);

double cross_entropy_loss(vector* pred, vector* real);

double compute(Layer_t* l, Index cur, Layer_t* prev, double(*a)(double));

void softmax(Layer_t* in);

void compute_next(Layer_t* prev, Layer_t* cur, double(*activation1)(double));

void forwards(Layer_t* input, Layer_t* hidden1, Layer_t* hidden2, Layer_t* hidden3, Layer_t* output);

void backwards(Layer_t* input, Layer_t* hidden1, Layer_t* hidden2, Layer_t* hidden3, Layer_t* output, vector* real, double rate);

Layer_t* lt_create(Index len, char weights, Index prev_len, vector* input);

#endif