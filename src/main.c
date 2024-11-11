#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linear_algebra_stuff/matrix_stuff/matrix.h"
#include "linear_algebra_stuff/vector_stuff/vector.h"
#include "neural_network_stuff/neural_structions.h"

void matrix_stuff(void) {
    matrix *m1 = m_create(6, 6, NULL);
    matrix *m2 = m_create(6, 6, NULL);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            m1->values[j][i] = rand()/444.0;
            m2->values[i][j] = i + j;
        }
    }

    printf(m_to_string(m_inverse(m1)));
    return;
}

void vector_stuff(void) {
    double test1[3] = { -6, 1, 3 };
    double test2[3] = { 6, -1, -3 };

    double test1_1[3] = { 1, 2, 3 };
    double test1_2[3] = { 4, 5, 6 };
    double test1_3[3] = { 7, 8, 9 };

    double *test1_m[3] = { test1_1, test1_2, test1_3 };

    vector *v1 = v_create(sizeof(test1) / sizeof(double), test1);
    vector *v2 = v_create(sizeof(test2) / sizeof(double), test2);
    matrix *m = m_create(3, 3, test1_m);

    vector *v_m_mult = m_v_mult(m, v1);
    vector *v_sig = v_sigmoid(v_m_mult);
    vector *v1_v2_add = v_add(v1, v2);

    // printf(v_to_string(v1_v2_add));

    return;
}

int main(int *argc, char *argv[]) {
    matrix_stuff();
    vector_stuff();

    return 0;
}