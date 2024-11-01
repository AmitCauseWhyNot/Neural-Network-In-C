#include "../matrix_stuff/matrix.h"

typedef unsigned int Index;

typedef struct {
    Index length;
    double *values;
} vector;

char* v_to_string(vector *v);

double* r_scale(double *row, double scale, Index size);

double r_sum(double *row, Index size);

// sigmoid = 1/(1 + e^-x)
double sigmoid(double value);

vector* v_sigmoid(vector *v);

vector* v_create(Index length, double *values);

vector* v_scale(vector *v, double scale);

vector* v_add(vector *v1, vector *v2);

vector* m_v_mult(matrix *m, vector *v);