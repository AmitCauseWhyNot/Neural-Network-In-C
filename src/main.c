// BIG BIG IMPORTANT

// STOPPED AT WHERE THE NAN STARTS. PROBABLY DOT PRODUCT OR SMTH LIKE THAT IDFK GOOD LUCK NIGGA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linear_algebra_stuff/matrix_stuff/matrix.h"
#include "linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_network_stuff/neural_structures.h"
#include "MNIST_stuff/mnist.h"

#define SCALE 8192

matrix *W1, *W2, *W3;
vector *b1, *b2, *b3, *Z1, *Z2, *Z3, *A1, *A2, *A3;

char weight_path[3][34] = {"./src/parameter_stuff/weight1.txt", "./src/parameter_stuff/weight2.txt", "./src/parameter_stuff/weight3.txt"};
char biases_path[3][32] = {"./src/parameter_stuff/bias1.txt", "./src/parameter_stuff/bias2.txt", "./src/parameter_stuff/bias3.txt"};

void set_parameters(void)
{
    W1 = m_create(784, 16, NULL);
    W2 = m_create(16, 16, NULL);
    W3 = m_create(16, 10, NULL);

    b1 = v_create(16, NULL);
    b2 = v_create(16, NULL);
    b3 = v_create(10, NULL);

    srand(time(NULL));
    for (int i = 0; i < 16; i++)
    {
        b1->values[i] = (double)rand() / SCALE;
        b2->values[i] = (double)rand() / SCALE;
        if (i < 10)
        {
            b3->values[i] = (double)rand() / SCALE;
        }
    }

    srand(time(NULL));
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            W2->values[i][j] = (double)rand() / SCALE;
            if (j < 10)
            {
                W3->values[i][j] = (double)rand() / SCALE;
            }
        }
    }

    srand(time(NULL));
    for (int i = 0; i < 784; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            W1->values[i][j] = (double)rand() / RAND_MAX;
        }
    }

    return;
}

void save_parameters(void)
{
    FILE *f;

    f = fopen(weight_path[0], "w");
    for (int i = 0; i < W1->Nrows; i++)
    {
        for (int j = 0; j < W1->Ncols; j++)
        {
            fprintf(f, "%.6f\n", W1->values[i][j]);
        }
    }
    fclose(f);

    f = fopen(weight_path[1], "w");
    for (int i = 0; i < W2->Nrows; i++)
    {
        for (int j = 0; j < W2->Ncols; j++)
        {
            fprintf(f, "%.6f\n", W2->values[i][j]);
        }
    }
    fclose(f);

    f = fopen(weight_path[2], "w");
    for (int i = 0; i < W3->Nrows; i++)
    {
        for (int j = 0; j < W3->Ncols; j++)
        {
            fprintf(f, "%.6f\n", W3->values[i][j]);
        }
    }
    fclose(f);

    f = fopen(biases_path[0], "w");
    for (int i = 0; i < b1->length; i++)
    {
        fprintf(f, "%.6f\n", b1->values[i]);
    }
    fclose(f);

    f = fopen(biases_path[1], "w");
    for (int i = 0; i < b2->length; i++)
    {
        fprintf(f, "%.6f\n", b2->values[i]);
    }
    fclose(f);

    f = fopen(biases_path[2], "w");
    for (int i = 0; i < b3->length; i++)
    {
        fprintf(f, "%.6f\n", b3->values[i]);
    }
    fclose(f);

    return;
}

void forward_propagation(vector *input)
{
    Z1 = v_add(m_v_mult(m_transpose(W1), input), b1);
    A1 = v_relu(Z1);

    Z2 = v_add(m_v_mult(m_transpose(W2), A1), b2);
    A2 = v_relu(Z2);

    Z3 = v_add(m_v_mult(m_transpose(W3), A2), b3);
    A3 = softmax(Z3);

    return;
}

void backward_propagation(double learning_rate, vector *real_value, vector *input_layer)
{
    vector *db3 = get_output_lambda(A3, real_value);
    vector *db2 = get_hidden_lambda(W3, db3, Z2);
    vector *db1 = get_hidden_lambda(W2, db2, Z1);

    matrix *dW3 = m_scale(v_vT_mult(db3, A2), learning_rate);
    matrix *dW2 = m_scale(v_vT_mult(db2, A1), learning_rate);
    matrix *dW1 = m_scale(v_vT_mult(db1, input_layer), learning_rate);

    W3 = m_sub(W3, dW3);
    b3 = v_sub(b3, db3);
    m_free(dW3);
    v_free(db3);

    W2 = m_sub(W2, dW2);
    b2 = v_sub(b2, db2);
    m_free(dW2);
    v_free(db2);

    W1 = m_sub(W1, dW1);
    b1 = v_sub(b1, db1);
    m_free(dW1);
    v_free(db1);

    return;
}

void train(double learning_rate, double stop)
{
    char image_train_path[] = "./src/data_stuff/train-images.idx3-ubyte";
    char label_train_path[] = "./src/data_stuff/train-labels.idx1-ubyte";

    uint8_t buffer[4];
    int num_images, label;
    double loss;
    vector *image;
    vector *real;

    FILE *f;
    f = fopen(image_train_path, "rb");

    if (!f)
    {
        perror("Cannot open image training file");
        return;
    }

    fseek(f, 4, SEEK_CUR);
    if (fread(buffer, sizeof(uint8_t), 4, f) != 4)
    {
        perror("Error reading from file");
        fclose(f);
        return;
    }
    num_images = convert_to_32int(buffer);

    set_parameters();
    for (int i = 0; i < num_images; i++)
    {
        real = v_create(10, NULL);
        image = v_create(784, get_image(image_train_path, i));
        label = get_label(label_train_path, i);
        real->values[label] = 1.0;

        forward_propagation(image);
        backward_propagation(learning_rate, real, image);
        save_parameters();

        loss = loss_function(A3, real);
        printf("[DONE] [WRITE] Epoch %d, loss: %.6f\n", i + 1, loss);
        if (loss <= stop)
        {
            printf("Bro reached heaven :O (diverged)\n");
            break;
        }

        v_free(real);
        v_free(image);
    }

    return;
}

int main(int argc, char **argv)
{
    train(0.01, (double)1e-6);

    return 0;
}
