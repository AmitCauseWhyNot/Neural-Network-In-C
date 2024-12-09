#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include "linear_algebra_stuff/matrix_stuff/matrix.h"
#include "linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_network_stuff/neural_structions.h"

#define SCALE 0.10

float getRandomFloat(float min, float max)
{
    return min + (float)rand() / RAND_MAX * (max - min);
}

void matrix_stuff(void)
{
    matrix *m1 = m_create(784, 16, NULL);
    matrix *m2 = m_create(16, 784, NULL);

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            m1->values[j][i] = i * j;
            m2->values[i][j] = i + j;
        }
    }

    double *row = m1->values[2];
    return;
}

void vector_stuff(void)
{
    double test1[3] = {-6, 1, 3};
    double test2[3] = {6, -1, -3};

    double test1_1[3] = {1, 2, 3};
    double test1_2[3] = {4, 5, 6};
    double test1_3[3] = {7, 8, 9};

    double *test1_m[3] = {test1_1, test1_2, test1_3};

    vector *v1 = v_create(sizeof(test1) / sizeof(double), test1);
    vector *v2 = v_create(sizeof(test2) / sizeof(double), test2);
    matrix *m = m_create(3, 3, test1_m);

    vector *v_m_mult = m_v_mult(m, v1);
    vector *v_sig = v_sigmoid(v_m_mult);
    vector *v1_v2_add = v_add(v1, v2);

    return;
}

void neural_structions_stuff(void)
{

    Layer *input_l = malloc(sizeof(Layer));
    Layer *hidden_l = malloc(sizeof(Layer));

    input_l->size = 784;
    input_l->neurons = malloc(input_l->size * sizeof(Neuron));

    hidden_l->size = 16;
    hidden_l->neurons = malloc(hidden_l->size * sizeof(Neuron));

    for (int i = 0; i < input_l->size; i++)
    {
        input_l->neurons[i].value = SCALE / getRandomFloat(10.0, 13.0);
        input_l->neurons[i].bias = SCALE / getRandomFloat(1.0, 14.0);
        input_l->neurons[i].size_w = hidden_l->size;
        input_l->neurons[i].weights = malloc(input_l->neurons[i].size_w * sizeof(double));
        for (int j = 0; j < input_l->neurons[i].size_w; j++)
        {
            input_l->neurons[i].weights[j] = SCALE / getRandomFloat(12.0, 15.0);
        }
    }

    for (int i = 0; i < hidden_l->size; i++)
    {
        hidden_l->neurons[i].bias = SCALE / getRandomFloat(5.0, 8.0);
        hidden_l->neurons[i].value = next_neuron_value(input_l, hidden_l, i);
        printf("%f\n", hidden_l->neurons[i].value);
    }

    return;
}

int main(int argc, char **argv)
{
    int size = 4;
    srand(time(NULL));

    matrix *m1 = m_create(size, size, NULL);
    matrix *m2 = m_create(size, size, NULL);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            m1->values[i][j] = rand() / 10000.0;
            m2->values[i][j] = rand() / 10000.0;
        }
    }

    printf(m_to_string(m_mult(m1, m2)));
    // printf(m_to_string(m_mult(m1, m2)));

    return 0;
}
