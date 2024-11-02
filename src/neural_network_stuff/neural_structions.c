#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../linear_algebra_stuff/matrix_stuff/matrix.h"
#include "../linear_algebra_stuff/vector_stuff/vector.h"

double sigmoid(double value) {
    return 1.0 / (1 + exp(value));
}

vector* v_sigmoid(vector *v) {
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++) {
        v_return->values[i] = sigmoid(v->values[i]);
    }

    return v_return;
}