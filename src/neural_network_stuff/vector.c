#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../matrix_stuff/matrix.h"
#include "vector.h"

char* v_to_string(vector *v) {
    int buf_size = v->length * 10 + v->length * 2 + v->length / 10 * 2 + 20;
    char *result = malloc(buf_size);
    if (result == NULL) {
        return NULL;
    }

    char *ptr = result;
    ptr += sprintf(ptr, "[\n   Length: %u,\n   [", v->length);

    for (Index i = 0; i < v->length; i++) {
        ptr += sprintf(ptr, "%.4f", v->values[i]);

        if (i < v->length - 1) {
            ptr += sprintf(ptr, ", ");
        }
        
        if ((i + 1) % 10 == 0 && i < v->length - 1) {
            ptr += sprintf(ptr, "\n    ");
        }
    }

    sprintf(ptr, "]\n]\n");

    return result;
}

double* r_scale(double *row, double scale, Index size) {
    double *r_return = malloc(size * sizeof(double));
    
    for (int i = 0; i < size; i++) {
        r_return[i] = row[i] * scale;
    }

    return r_return;
}

double r_sum(double *row, Index size) {
    double sum;

    for (int i = 0; i < size; i++) {
        sum += row[i];
    }

    return sum;
}

double sigmoid(double value) {
    return 1.0/(1.0 + exp(-1.0 * value));
}

vector* v_sigmoid(vector *v) {
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++) {
        v_return->values[i] = sigmoid(v->values[i]);
    }

    return v_return;
}

vector* v_create(Index length, double *values) {
    vector* v_return = malloc(sizeof(vector));

    if (v_return == NULL) {
        return NULL;
    }

    v_return->length = length;
    v_return->values = malloc(length * sizeof(double));

    if (v_return->values == NULL) {
        return NULL;
    }

    if (values != NULL) {
        v_return->values = values;
    }

    return v_return;
}

vector* v_scale(vector *v, double scale) {
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++) {
        v_return->values[i] = v->values[i] * scale;
    }

    return v_return;
}

vector* v_add(vector *v1, vector *v2) {
    if (v1->length != v2->length) {
        return NULL;
    }

    vector *v_return = v_create(v1->length, NULL);

    for (int i = 0; i < v_return->length; i++) {
        v_return->values[i] = v1->values[i] + v2->values[i];
    }

    return v_return;
}

vector* m_v_mult(matrix* m, vector* v) {
    if (m->Nrows != v->length) {
        return NULL;
    }
    
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v->length; i++) {
        v_return->values[i] = m_dot(m->values[i], v->values, v->length);
    }

    return v_return;
}

