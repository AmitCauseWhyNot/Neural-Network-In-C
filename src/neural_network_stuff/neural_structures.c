#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "neural_structures.h"

void n_free(Neuron_t* n)
{
    free(n);
}

void l_free(Layer_t* l)
{
    for (int i = 0; i < l->len; i++)
    {
        n_free(l->neurons[i]);
    }

    free(l->neurons);
    free(l);
}

vector* get_values_vector(Layer_t* l)
{
    vector* v = v_create(l->len, NULL);
    for (Index i = 0; i < l->len; ++i) v->values[i] = l->neurons[i]->value;
    return v;
}

vector* get_label_vector(Index lbl)
{
    vector* r = v_create(OUT_LENGTH, NULL);

    for (int i = 0; i < OUT_LENGTH; i++)
    {
        r->values[i] = (i == lbl) ? 1 : 0;
    }

    return r;
}

double leaky(double x)
{
    return (x < 0) ? LEAKY * x : x;
}

double d_leaky(double x)
{
    return (x < 0) ? LEAKY : 1;
}

vector* vd_leaky(vector* v)
{
    vector* r = v_create(v->length, NULL);
    for (Index i = 0; i < v->length; ++i)
        r->values[i] = d_leaky(v->values[i]);
    return r;
}

void update_bias(Layer_t* l, vector* offset, double rate)
{
    for(int i = 0; i < l->len; i++)
    {
        l->neurons[i]->bias -= rate * offset->values[i];
    }
}

void update_weights(Layer_t* l, matrix* offset, double rate)
{
    for (int i = 0; i < offset->Nrows; i++)
    {
        for (int j = 0; j < offset->Ncols; j++)
        {
            l->weights->values[i][j] -= rate * offset->values[i][j];
        }
    }
}

double cross_entropy_loss(vector* pred, vector* real)
{
    double loss = 0.0;

    for (int i = 0; i < pred->length; i++)
    {
        loss -= real->values[i] * log(pred->values[i] + 1e-12);
    }

    return loss;
}

double rand_uniform(double a, double b)
{
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

double rand_normal(double mean, double std) {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return mean + z * std;
}

double he_std(int fan_in)
{
    return sqrt(2.0 / fan_in);
}

double compute(Layer_t* l, Index cur_idx, Layer_t* prev, double(*a)(double))
{
    double sum = l->neurons[cur_idx]->bias;
    if (l->weights) {
        for (Index j = 0; j < prev->len; ++j) {
            sum += l->weights->values[cur_idx][j] * prev->neurons[j]->value;
        }
    }
    return a ? a(sum) : sum;
}

void softmax(Layer_t* in)
{
    double max = in->neurons[0]->value;
    double sum = 0.0;

    for (int i = 1; i < in->len; i++)
    {
        if (in->neurons[i]->value > max)
        {
            max = in->neurons[i]->value;
        }
    }

    for (int i = 0; i < in->len; i++)
    {
        sum += exp(in->neurons[i]->value - max);
    }

    for (int i = 0; i < in->len; i++)
    {
        in->neurons[i]->value = exp(in->neurons[i]->value - max) / sum;
    }
}

void compute_next(Layer_t* prev, Layer_t* cur, double(*activation1)(double))
{
    for (int i = 0; i < cur->len; i++)
    {
        cur->neurons[i]->value = compute(cur, i, prev, activation1);
    }

    if (!activation1)
    {
        softmax(cur);
    }
}

void forwards(Layer_t* input, Layer_t* hidden1, Layer_t* hidden2, Layer_t* hidden3, Layer_t* output)
{
    compute_next(input, hidden1, &leaky);
    compute_next(hidden1, hidden2, &leaky);
    compute_next(hidden2, hidden3, &leaky);
    compute_next(hidden3, output, NULL);
}

vector* compute_layer_delta(matrix* next_weights, vector* next_delta, vector* current_activations, vector* (*derivative_func)(vector*)) {
    matrix* weights_T = m_transpose(next_weights);
    vector* weights_delta = m_v_mult(weights_T, next_delta);
    vector* activation_deriv = derivative_func(current_activations);
    vector* layer_delta = H_product(weights_delta, activation_deriv);
    
    m_free(weights_T);
    v_free(weights_delta);
    v_free(activation_deriv);
    
    return layer_delta;
}

void update_layer(Layer_t* layer, vector* layer_delta, vector* prev_activations, double rate) {
    matrix* weight_grad = v_vT_mult(layer_delta, prev_activations);
    
    update_weights(layer, weight_grad, rate);
    update_bias(layer, layer_delta, rate);
    
    m_free(weight_grad);
}

void backwards(Layer_t* input, Layer_t* hidden1, Layer_t* hidden2, Layer_t* hidden3, Layer_t* output, vector* real, double rate) {
    
    vector* predicted = get_values_vector(output);
    vector* hidden3_values = get_values_vector(hidden3);
    vector* hidden2_values = get_values_vector(hidden2);
    vector* hidden1_values = get_values_vector(hidden1);
    vector* input_values = get_values_vector(input);

    // Output layer
    vector* out_delta = v_sub(predicted, real);
    update_layer(output, out_delta, hidden3_values, rate);
    
    // Hidden3 layer
    vector* hid3_delta = compute_layer_delta(output->weights, out_delta, hidden3_values, &vd_leaky);
    v_free(out_delta);
    update_layer(hidden3, hid3_delta, hidden2_values, rate);
    
    // Hidden2 layer
    vector* hid2_delta = compute_layer_delta(hidden3->weights, hid3_delta, hidden2_values, &vd_leaky);
    v_free(hid3_delta);
    update_layer(hidden2, hid2_delta, hidden1_values, rate);
    
    // Hidden1 layer
    vector* hid1_delta = compute_layer_delta(hidden2->weights, hid2_delta, hidden1_values, &vd_leaky);
    v_free(hid2_delta);
    update_layer(hidden1, hid1_delta, input_values, rate);
    v_free(hid1_delta);

    v_free(predicted);
    v_free(hidden3_values);
    v_free(hidden2_values);
    v_free(hidden1_values);
    v_free(input_values);
}

Layer_t* lt_create(Index len, char weights, Index prev_len, vector* input)
{
    Layer_t* r = malloc(sizeof(Layer_t));
    if (!r) return NULL;

    r->len = len;
    r->neurons = malloc(len * sizeof(Neuron_t*));
    if (!r->neurons) { free(r); return NULL; }

    for (Index i = 0; i < len; ++i) {
        r->neurons[i] = malloc(sizeof(Neuron_t));
        r->neurons[i]->value = 0.0;
        r->neurons[i]->bias = 0.0;
    }

    if (input) {
        for (Index i = 0; i < input->length && i < len; ++i)
            r->neurons[i]->value = input->values[i];
        return r;
    }

    // If this layer has incoming weights, allocate weights matrix with shape (len x prev_len)
    if (weights && prev_len > 0) {
        r->weights = m_create(len, prev_len, NULL);
        double std = he_std(prev_len); // fan_in = prev_len
        for (Index i = 0; i < len; ++i) {
            r->neurons[i]->bias = BASE_BIAS;
            for (Index j = 0; j < prev_len; ++j) {
                r->weights->values[i][j] = rand_normal(0.0, std);
            }
        }
    } else {
        r->weights = NULL;
    }

    return r;
}