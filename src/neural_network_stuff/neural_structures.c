#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_structures.h"

double sigmoid(double value)
{
    return 1.0 / (1 + exp(-1.0 * value));
}

double d_sigmoid(double value)
{
    return sigmoid(value) * (1 - sigmoid(value));
}

vector *softmax(vector *v)
{
    vector *v_return = v_create(v->length, NULL);

    double exp_sum = 0.0;

    for (int i = 0; i < v->length; i++)
    {
        exp_sum += exp(v->values[i]);
    }

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = exp(v->values[i]) / exp_sum;
    }

    return v_return;
}

double next_neuron_value(Layer *l, Layer *n_l, int index)
{
    double sum_values = 0.0;

    for (int i = 0; i < l->size; i++)
    {
        sum_values += (l->neurons[i].weights[index] * l->neurons[i].value);
        // printf("Index: %d, This Value: %f, Weight: %f, Value: %f\n", i, sum_values, l->neurons[i].weights[i] * l->neurons[i].value);
    }

    sum_values += n_l->neurons[index].bias;

    return sigmoid(sum_values);
}

double loss_function(vector *prediction, double *real)
{
    double total_loss = 0, value;

    for (int i = 0; i < prediction->length; i++)
    {
        value = (real[i] - prediction->values[i]);
        total_loss += (value * value);
    }

    return total_loss / (prediction->length * 2);
}

matrix *get_weight_gradient(vector *L, vector *A)
{
    return v_vT_mult(L, A);
}

vector *get_bias_gradient(vector *L)
{
    return L;
}

vector *get_hidden_lambda(matrix *W, vector *L, double (*a_f)(double), vector *Z)
{
    W = m_transpose(W);
    Z = v_d_sigmoid(Z);

    vector *l_return = v_create(W->Nrows, NULL);
    vector *W_L_mult = m_v_mult(W, L);

    l_return = H_product(W_L_mult, Z);

    return l_return;
}

vector *get_output_lambda(vector *A, vector *Y)
{
    if (A->length != Y->length)
    {
        return NULL;
    }

    return v_sub(A, Y);
}

vector *v_d_sigmoid(vector *v)
{
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = d_sigmoid(v->values[i]);
    }

    return v_return;
}

vector *v_sigmoid(vector *v)
{
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = sigmoid(v->values[i]);
    }

    return v_return;
}