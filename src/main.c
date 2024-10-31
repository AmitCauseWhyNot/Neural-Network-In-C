#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_stuff/matrix.h"

int main(int *argc, char *argv[]) {
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
    printf(m_to_string(div_m3_m4));
    // printf("%f", m_det(m3));
    
    return 0;
}