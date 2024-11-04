#ifndef VECTOR_H
#define VECTOR_H

#include "../matrix_stuff/matrix.h"

typedef struct {
    Index length;
    double *values;
} vector;

// Takes a vector *v and returns a string containing the values and length of the vector.
char* v_to_string(vector *v);

// Takes a pointer to a double array, a scale and the size of the array and returns a scaled array.
double* r_scale(double *row, double scale, Index size);

// Takes a pointer to an array and it's size and returns the sum of the values in the array.
double r_sum(double *row, Index size);

// Takes the variables length and a pointer to an array of values and returns a vector the size of the length and containing values.
vector* v_create(Index length, double *values);

// Takes a vector *v and a scale and returns a scaled vector.
vector* v_scale(vector *v, double scale);

// Takes a vector *v1 and a vector *v2 and returns the added vector of them.
vector* v_add(vector *v1, vector *v2);

// Takes a matrix *m and a vector *v and returns the multiplied vector of them.
vector* m_v_mult(matrix *m, vector *v);

#endif // VECTOR_H