#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "linear_algebra_stuff/matrix_stuff/matrix.h"
#include "linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_network_stuff/neural_structures.h"
#include "MNIST_stuff/mnist.h"

#define EPOCH_NUM 10           // Number of epochs (full passes through dataset)
#define TRAIN_SAMPLES 60000
#define IMG_LENGTH 784
#define OUT_LENGTH 10
#define HIDDEN1_LENGTH 256
#define HIDDEN2_LENGTH 128
#define HIDDEN3_LENGTH 64
#define BATCH_SIZE 32          // Mini-batch size
#define INITIAL_LR 0.01        // Initial learning rate
#define TEST_COUNT 10000

#define img_train_path "./src/data_stuff/train-images.idx3-ubyte"
#define lbl_train_path "./src/data_stuff/train-labels.idx1-ubyte"
#define img_test_path "./src/data_stuff/t10k-images.idx3-ubyte"
#define lbl_test_path "./src/data_stuff/t10k-labels.idx1-ubyte"

double get_learning_rate(int epoch) 
{
    if (epoch < 3) return INITIAL_LR;
    if (epoch < 6) return INITIAL_LR * 0.5;
    if (epoch < 8) return INITIAL_LR * 0.1;
    return INITIAL_LR * 0.05;
}

// Shuffle indices for randomized training
void shuffle_indices(int* indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

void set_values(double* a1, double* a2, int l) {
    for (int i = 0; i < l; i++) {
        a1[i] = a2[i];
    }
}

// Gradient clipping to prevent exploding gradients
void clip_vector(vector* v, double max_norm) {
    double norm = 0.0;
    for (int i = 0; i < v->length; i++) {
        norm += v->values[i] * v->values[i];
    }
    norm = sqrt(norm);
    
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < v->length; i++) {
            v->values[i] *= scale;
        }
    }
}

int main() {
    srand(time(NULL));
    
    Layer_t* hidden1 = lt_create(HIDDEN1_LENGTH, 1, IMG_LENGTH, NULL);
    Layer_t* hidden2 = lt_create(HIDDEN2_LENGTH, 1, HIDDEN1_LENGTH, NULL);
    Layer_t* hidden3 = lt_create(HIDDEN3_LENGTH, 1, HIDDEN2_LENGTH, NULL);
    Layer_t* output = lt_create(OUT_LENGTH, 1, HIDDEN3_LENGTH, NULL);

    vector* img = v_create(IMG_LENGTH, NULL);
    vector* v_lbl = v_create(OUT_LENGTH, NULL);

    int* indices = malloc(TRAIN_SAMPLES * sizeof(int));
    for (int i = 0; i < TRAIN_SAMPLES; i++) {
        indices[i] = i;
    }

    printf("Starting training with mini-batches (batch_size=%d)...\n", BATCH_SIZE);

    // Training loop - iterate over epochs
    for (int epoch = 1; epoch <= EPOCH_NUM; epoch++) {
        double lr = get_learning_rate(epoch);
        double epoch_loss = 0.0;
        int num_batches = 0;
        
        
        shuffle_indices(indices, TRAIN_SAMPLES);
        printf("Epoch %d/%d (LR=%.6f)\n", epoch, EPOCH_NUM, lr);
        
        // Process mini-batches
        for (int batch_start = 0; batch_start < TRAIN_SAMPLES; batch_start += BATCH_SIZE) {
            double batch_loss = 0.0;
            int current_batch_size = BATCH_SIZE;
            
            if (batch_start + BATCH_SIZE > TRAIN_SAMPLES) {
                current_batch_size = TRAIN_SAMPLES - batch_start;
            }
            
            // Process each sample in the batch
            for (int i = 0; i < current_batch_size; i++) {
                int sample_idx = indices[batch_start + i];
                
                double* cur_img = get_image(img_train_path, sample_idx);
                if (!cur_img) {
                    printf("ERROR: Failed to load image at index %d\n", sample_idx);
                    continue;
                }
                
                vector* cur_v_lbl = get_label_vector(get_label(lbl_train_path, sample_idx));
                if (!cur_v_lbl) {
                    printf("ERROR: Failed to load label at index %d\n", sample_idx);
                    free(cur_img);
                    continue;
                }
                
                set_values(img->values, cur_img, IMG_LENGTH);
                set_values(v_lbl->values, cur_v_lbl->values, OUT_LENGTH);
                
                Layer_t* input = lt_create(IMG_LENGTH, 0, 0, img);
                
                forwards(input, hidden1, hidden2, hidden3, output);
                
                vector* pred = get_values_vector(output);
                batch_loss += cross_entropy_loss(pred, v_lbl);
                v_free(pred);
                
                // Backward pass with scaled learning rate
                backwards(input, hidden1, hidden2, hidden3, output, v_lbl, lr / current_batch_size);
                
                l_free(input);
                v_free(cur_v_lbl);
                free(cur_img);
            }
            
            // Track statistics
            epoch_loss += batch_loss;
            num_batches++;
            
            // Print progress every 500 batches
            if (num_batches % 500 == 0) {
                printf("  Batch %d/%d: Avg Loss = %.6f\n", 
                       num_batches, 
                       (TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE,
                       batch_loss / current_batch_size);
            }
        }
        
        // Print epoch summary
        printf("  Epoch %d Average Loss: %.6f\n\n", epoch, epoch_loss / TRAIN_SAMPLES);
    }

    printf("\nTraining complete! Starting evaluation...\n");
    
    v_free(v_lbl);
    int count_correct = 0;

    int test_start_idx = 0;
    for (int i = test_start_idx; i < test_start_idx + TEST_COUNT; i++) {
        double* cur_img = get_image(img_test_path, i);
        if (!cur_img) {
            printf("ERROR: Failed to load test image %d\n", i);
            continue;
        }
        
        int cur_lbl = get_label(lbl_test_path, i);

        set_values(img->values, cur_img, IMG_LENGTH);
        Layer_t* input = lt_create(IMG_LENGTH, 0, 0, img);

        forwards(input, hidden1, hidden2, hidden3, output);

        // Find predicted class
        double max = output->neurons[0]->value;
        int max_index = 0;
        for (int k = 1; k < OUT_LENGTH; k++) {
            if (output->neurons[k]->value > max) {
                max = output->neurons[k]->value;
                max_index = k;
            }
        }

        if (max_index == cur_lbl) {
            count_correct++;
        }

        l_free(input);
        free(cur_img);
    }

    printf("\nTest Accuracy: %.2f%% (%d/%d correct)\n", (double)count_correct / (double)TEST_COUNT * 100.0, count_correct, TEST_COUNT);

    free(indices);
    v_free(img);
    l_free(hidden1);
    l_free(hidden2);
    l_free(output);

    return 0;
}