#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix_stuff/matrix.h"
#include "neural_network_stuff/vector.h"

void matrix_stuff(void) {
    double subtest1_1[2] = { 1, 2 };
    double subtest1_2[2] = { 3, 4 };

    double subtest2_1[2] = { 5, 6 };
    double subtest2_2[2] = { 7, 8 };

    double subtest3_1[3] = { 5, 3, 2 };
    double subtest3_2[3] = { 1, 10, 9 };
    double subtest3_3[3] = { 34, 247, 103 };

    double subtest4_1[3] = { 0, 8, 88 };
    double subtest4_2[3] = { 75, 42, 29 };
    double subtest4_3[3] = { 84, 94, 104 };

    double *test1[2] = { subtest1_1, subtest1_2 };
    double *test2[2] = { subtest2_1, subtest2_2 };
    double *test3[3] = { subtest3_1, subtest3_2, subtest3_3 };
    double *test4[3] = { subtest4_1, subtest4_2, subtest4_3 };

    matrix *m1 = m_create(2, 2, test1);
    matrix *m2 = m_create(2, 2, test2);
    matrix *m3 = m_create(3, 3, test3);
    matrix *m4 = m_create(3, 3, test4);
    matrix *mT = m_transpose(m2);
    matrix *added = m_add(m1, m2);
    matrix *sub_m3 = m_get_sub(m3, 1, 1);
    matrix *mult_m3_m4 = m_mult(m3, m4);
    matrix *inverse_m3 = m_inverse(m3);
    matrix *div_m3_m4 = m_div(m3, m4);

    // printf(m_to_string(*m1));
    // printf(m_to_string(*m2));
    // printf(m_to_string(*m3));
    // printf(m_to_string(*mT));
    // printf(m_to_string(*added));
    // printf(m_to_string(*sub_m3));
    // printf(m_to_string(mult_m3_m4));
    // printf(m_to_string(inverse_m3));
    // printf(m_to_string(div_m3_m4));
    // printf("%f", m_det(m3));

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

    // printf(v_to_string(v_m_mult));
    // printf(v_to_string(v_sig));
    printf(v_to_string(v1_v2_add));

    return;
}

int main(int *argc, char *argv[]) {
    matrix_stuff();
    vector_stuff();

    return 0;
}