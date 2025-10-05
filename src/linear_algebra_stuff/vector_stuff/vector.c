#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../matrix_stuff/matrix.h"
#include "vector.h"

char *v_to_string(vector *v)
{
    int buf_size = v->length * 10 + v->length * 2 + v->length / 10 * 2 + 20;
    char *result = malloc(buf_size);
    if (result == NULL)
    {
        return NULL;
    }

    char *ptr = result;
    ptr += sprintf(ptr, "[\n   Length: %u,\n   [", v->length);

    for (Index i = 0; i < v->length; i++)
    {
        ptr += sprintf(ptr, "%.4f", v->values[i]);

        if (i < v->length - 1)
        {
            ptr += sprintf(ptr, ", ");
        }

        if ((i + 1) % 10 == 0 && i < v->length - 1)
        {
            ptr += sprintf(ptr, "\n    ");
        }
    }

    sprintf(ptr, "]\n]\n");

    return result;
}

double *r_scale(double *row, double scale, Index size)
{
    double *r_return = malloc(size * sizeof(double));

    for (int i = 0; i < size; i++)
    {
        r_return[i] = row[i] * scale;
    }

    return r_return;
}

double r_sum(double *row, Index size)
{
    double sum = 0.0;
    for (Index i = 0; i < size; ++i) sum += row[i];
    return sum;
}

vector *v_create(Index length, double *values)
{
    vector *v_return = malloc(sizeof(vector));
    if (!v_return) { perror("v_create"); return NULL; }

    v_return->length = length;
    v_return->values = malloc(length * sizeof(double));
    if (!v_return->values) { perror("v_create values"); free(v_return); return NULL; }

    // initialize to zero
    for (Index i = 0; i < length; ++i) v_return->values[i] = 0.0;

    // if values supplied, copy them in
    if (values != NULL) {
        for (Index i = 0; i < length; ++i) v_return->values[i] = values[i];
    }

    return v_return;
}

void v_free(vector *v)
{
    if (!v) return;
    free(v->values);
    free(v);
}

vector *v_scale(vector *v, double scale)
{
    vector *v_return = v_create(v->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = v->values[i] * scale;
    }

    return v_return;
}

vector *v_add(vector *v1, vector *v2)
{
    if (v1->length != v2->length)
    {
        fprintf(stderr, "v_add ERROR: lengths not the same\n");
        fprintf(stderr, "v1 length: %d, v2 length: %d\n", v1->length, v2->length);
        return NULL;
    }

    vector *v_return = v_create(v1->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = v1->values[i] + v2->values[i];
    }

    return v_return;
}

vector *v_sub(vector *v1, vector *v2)
{
    if (v1->length != v2->length)
    {
        fprintf(stderr, "v_sub ERROR: lengths not the same\n");
        fprintf(stderr, "v1 length: %d, v2 length: %d\n", v1->length, v2->length);
        return NULL;
    }

    vector *v_return = v_create(v1->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = v1->values[i] - v2->values[i];
    }

    return v_return;
}

vector *m_v_mult(matrix *m, vector *v)
{
    if (m->Ncols != v->length)
    {
        fprintf(stderr, "m_v_mult ERROR: columns not equal to vector length\n");
        fprintf(stderr, "Matrix columns: %d, Vector length: %d\n", m->Ncols, v->length);
        return NULL;
    }

    vector *v_return = v_create(m->Nrows, NULL);

    for (int i = 0; i < m->Nrows; i++)
    {
        double dot_product = 0.0;
        for (int j = 0; j < m->Ncols; j++)
        {
            dot_product += m->values[i][j] * v->values[j];
        }
        v_return->values[i] = dot_product;
    }

    return v_return;
}

vector *v_m_weighted_sum(matrix *m, vector *v)
{
    if (m->Ncols != v->length)
    {
        return NULL;
    }

    vector *v_return = v_create(m->Nrows, NULL);

    for (int i = 0; i < m->Nrows; i++)
    {
        for (int j = 0; j < m->Ncols; j++)
        {
            v_return->values[i] += m->values[i][j] * v->values[j];
        }
    }

    return v_return;
}

vector *H_product(vector *v1, vector *v2)
{
    if (v1->length != v2->length)
    {
        fprintf(stderr, "H_product ERROR: vector length un-equal\n");
        fprintf(stderr, "v1 length: %d, v2 length: %d\n", v1->length, v2->length);
        return NULL;
    }

    vector *v_return = v_create(v1->length, NULL);

    for (int i = 0; i < v_return->length; i++)
    {
        v_return->values[i] = v1->values[i] * v2->values[i];
    }

    return v_return;
}

matrix *v_vT_mult(vector *v1, vector *v2)
{
    matrix *m_return = m_create(v1->length, v2->length, NULL);

    for (int i = 0; i < m_return->Nrows; i++)
    {
        for (int j = 0; j < m_return->Ncols; j++)
        {
            m_return->values[i][j] = v1->values[i] * v2->values[j];
        }
    }

    return m_return;
}
