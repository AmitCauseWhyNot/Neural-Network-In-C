#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_structures.h"

#define max(x, y) (((x) >= (y)) ? (x) : (y))

double relu(double value)
{
    return max(0, value);
}

double d_relu(double value)
{
    return (value <= 0) ? 0.0 : 1.0;
}

vector *softmax(vector *v)
{
    vector *v_return = v_create(v->length, NULL);

    // Find the maximum value
    double max_val = v->values[0];
    for (int i = 1; i < v->length; i++)
    {
        if (v->values[i] > max_val)
        {
            max_val = v->values[i];
        }
    }

    // Compute the sum of exponentials
    double exp_sum = 0.0;
    for (int i = 0; i < v->length; i++)
    {
        exp_sum += exp(v->values[i] - max_val);
    }

    // Add epsilon to prevent division by zero
    if (exp_sum < 1e-12)
    {
        exp_sum = 1e-12;
    }

    // Compute softmax probabilities
    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = exp(v->values[i] - max_val) / exp_sum;
    }

    return v_return;
}

double loss_function(vector *prediction, vector *real)
{
    if (prediction->length != real->length)
    {
        fprintf(stderr, "Loss function ERROR: vector lengths are not the same!\n");
        fprintf(stderr, "Prediction length: %d, Real length: %d\n", prediction->length, real->length);
        return -1.0;
    }

    double total_loss = 0, value;

    for (int i = 0; i < prediction->length; i++)
    {
        value = (real->values[i] - prediction->values[i]);
        total_loss += (value * value);
    }

    return total_loss / prediction->length;
}

matrix *get_weight_gradient(vector *L, vector *A)
{
    return v_vT_mult(L, A);
}

vector *get_hidden_lambda(matrix *W, vector *L, vector *Z)
{
    Z = v_d_relu(Z);

    vector *l_return = v_create(W->Nrows, NULL);
    vector *W_L_mult = m_v_mult(W, L);

    l_return = H_product(W_L_mult, Z);

    return l_return;
}

vector *get_output_lambda(vector *A, vector *Y)
{
    if (A->length != Y->length)
    {
        perror("get_output_lambda error: A length not equal to Y length\n");
        printf("A length: %d, Y length: %d\n", A->length, Y->length);
        return NULL;
    }

    return v_sub(A, Y);
}

vector *v_d_relu(vector *v)
{
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = d_relu(v->values[i]);
    }

    return v_return;
}

vector *v_relu(vector *v)
{
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = relu(v->values[i]);
    }

    return v_return;
}