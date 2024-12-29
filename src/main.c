#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include "linear_algebra_stuff/matrix_stuff/matrix.h"
#include "linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_network_stuff/neural_structures.h"
#include "MNIST_stuff/mnist.h"

matrix *W1, *W2, *W3;
vector *b1, *b2, *b3, *Z1, *Z2, *Z3, *A1, *A2, *A3;

void set_parameters(void)
{
    W1 = m_create(784, 16, NULL);
    W2 = m_create(16, 16, NULL);
    W3 = m_create(16, 10, NULL);

    b1 = v_create(16, NULL);
    b2 = v_create(16, NULL);
    b3 = v_create(10, NULL);

    for (int i = 0; i < 16; i++)
    {
        b1->values[i] = rand() / RAND_MAX;
        b2->values[i] = rand() / RAND_MAX;
        if (i < 10)
        {
            b3->values[i] = rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            W2->values[i][j] = rand() / RAND_MAX;
            if (j < 10)
            {
                W3->values[i][j] = rand() / RAND_MAX;
            }
        }
    }

    for (int i = 0; i < 784; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            W1->values[i][j] = rand() / RAND_MAX;
        }
    }

    return;
}

void forward_propagation(vector *input)
{
    Z1 = v_add(m_v_mult(m_transpose(W1), input), b1);
    A1 = v_sigmoid(Z1);

    Z2 = v_add(m_v_mult(W2, A1), b2);
    A2 = v_sigmoid(Z2);

    Z3 = v_add(m_v_mult(W3, A2), b3);
    A3 = softmax(Z3);
    return;
}

void backward_propagation(double learning_rate, vector *real_value)
{
    vector *db3 = get_output_lambda(A3, real_value);
    vector *db2 = get_hidden_lambda(W3, db3, &sigmoid, Z2);
    vector *db1 = get_hidden_lambda(W2, db2, &sigmoid, Z1);
    return;
}

int main(int argc, char **argv)
{
    char image_train_path[] = "./src/data_stuff/train-images.idx3-ubyte";
    char label_train_path[] = "./src/data_stuff/train-labels.idx1-ubyte";
    int label = get_label(label_train_path, 1);
    vector *actual_value = v_create(10, NULL);

    for (int i = 0; i < 10; i++)
    {
        if (i == label)
        {
            actual_value->values[i] = !actual_value->values[i];
            break;
        }
    }

    return 0;
}
