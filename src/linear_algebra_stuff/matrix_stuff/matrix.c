#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"

#define max(x, y) (((x) >= (y)) ? (x) : (y))
#define min(x, y) (((x) <= (y)) ? (x) : (y))

void m_free(matrix *m)
{
    if (!m) return;
    for (int i = 0; i < m->Nrows; i++)
    {
        free(m->values[i]);
    }
    free(m->values);
    free(m);
}

double m_det(matrix *m)
{
    if (m->Nrows != m->Ncols)
    {
        return -1.0;
    }
    else if (m->Nrows == 1)
    {
        return m->values[0][0];
    }

    double sum = 0, co = 0;
    for (int j = 0; j < m->Ncols; j++)
    {
        co = (double)(pow(-1, j));
        sum += co * m->values[0][j] * m_det(m_get_sub(m, 0, j));
    }

    return sum;
}

double m_dot(double *row, double *col, int size)
{
    double sum = 0.0;

    for (int i = 0; i < size; i++)
    {
        sum += row[i] * col[i];
    }

    return sum;
}

double *m_get_row(matrix *m, Index i)
{
    double *a_return = malloc(m->Ncols * sizeof(double));

    for (int j = 0; j < m->Ncols; j++)
    {
        a_return[j] = m->values[i][j];
    }

    return a_return;
}

double *m_get_col(matrix *m, Index j)
{
    double *a_return = malloc(m->Nrows * sizeof(double));

    for (int i = 0; i < m->Nrows; i++)
    {
        a_return[i] = m->values[i][j];
    }

    return a_return;
}

matrix *m_create(Index rows, Index cols, double **data)
{
    // Allocate memory for the matrix structure
    matrix *m = malloc(sizeof(matrix));
    if (m == NULL)
    {
        return NULL;
    }

    // Set the dimensions
    m->Nrows = rows;
    m->Ncols = cols;

    // If data is provided, use it directly and skip allocation
    if (data != NULL)
    {
        m->values = data;
        return m;
    }

    // Allocate memory for the row pointers
    m->values = malloc(rows * sizeof(double *));
    if (m->values == NULL)
    {
        free(m);
        return NULL;
    }

    // Allocate memory for each row
    for (int i = 0; i < rows; i++)
    {
        m->values[i] = malloc(cols * sizeof(double));
        if (m->values[i] == NULL)
        {
            // Free previously allocated rows in case of failure
            for (int j = 0; j < i; j++)
            {
                free(m->values[j]);
            }
            free(m->values);
            free(m);
            return NULL;
        }
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            m->values[i][j] = 0.0;
        }
    }

    return m;
}

matrix *m_scale(matrix *m, double scale)
{
    matrix *m_return = m_create(m->Nrows, m->Ncols, NULL);

    for (int i = 0; i < m_return->Nrows; i++)
    {
        for (int j = 0; j < m_return->Ncols; j++)
        {
            m_return->values[i][j] = m->values[i][j] * scale;
        }
    }

    return m_return;
}

matrix *m_transpose(matrix *m)
{
    matrix *m_return = m_create(m->Ncols, m->Nrows, NULL);

    for (int i = 0; i < m->Nrows; i++)
    {
        for (int j = 0; j < m->Ncols; j++)
        {
            m_return->values[j][i] = m->values[i][j];
        }
    }

    return m_return;
}

matrix *m_cofactor(matrix *m)
{
    matrix *m_return = m_create(m->Nrows, m->Ncols, NULL);

    for (int i = 0; i < m_return->Nrows; i++)
    {
        for (int j = 0; j < m_return->Ncols; j++)
        {
            m_return->values[i][j] = pow(-1, i + j) * m_det(m_get_sub(m, i, j));
        }
    }

    return m_return;
}

matrix *m_get_sub(matrix *m, Index row, Index col)
{
    matrix *m_return = m_create(m->Nrows - 1, m->Ncols - 1, NULL);
    double v_arr[(m->Nrows - 1) * (m->Ncols - 1)];
    int counter = 0;

    for (int i = 0; i < m->Nrows; i++)
    {
        for (int j = 0; j < m->Ncols; j++)
        {
            if (i != row && j != col)
            {
                v_arr[counter++] = m->values[i][j];
            }
        }
    }

    counter = 0;
    for (int i = 0; i < sizeof(v_arr) / sizeof(double); i++)
    {
        if (i % m_return->Nrows == 0 && i > 0)
        {
            m_return->values[++counter][i % m_return->Nrows] = v_arr[i];
        }
        else
        {
            m_return->values[counter][i % m_return->Nrows] = v_arr[i];
        }
    }

    return m_return;
}

matrix *m_adj(matrix *m)
{
    matrix *m_return = m_create(m->Ncols, m->Nrows, m_transpose(m_cofactor(m))->values);
    return m_return;
}

matrix *m_inverse(matrix *m)
{
    double det = m_det(m);

    if (det == 0)
    {
        return NULL;
    }

    matrix *almost_m_return = m_create(m->Nrows, m->Ncols, m_adj(m)->values);
    matrix *m_return = m_scale(almost_m_return, 1.0 / m_det(m));

    return m_return;
}

matrix *m_add(matrix *m1, matrix *m2)
{
    if (m1->Nrows != m2->Nrows || m1->Ncols != m2->Ncols)
    {
        return NULL;
    }

    matrix *m_return = m_create(m1->Nrows, m1->Ncols, NULL);

    for (int i = 0; i < m1->Nrows; i++)
    {
        for (int j = 0; j < m1->Ncols; j++)
        {
            m_return->values[i][j] = m1->values[i][j] + m2->values[i][j];
        }
    }

    return m_return;
}

matrix *m_sub(matrix *m1, matrix *m2)
{
    if (m1->Nrows != m2->Nrows || m1->Ncols != m2->Ncols)
    {
        fprintf(stderr, "m_sub ERROR: m1 and m2 not of equal shapes\n");
        fprintf(stderr, "m1 Nrows: %d, m1 Ncols: %d\nm2 Nrows: %d, m2 Ncols: %d", m1->Nrows, m1->Ncols, m2->Nrows, m2->Ncols);
        return NULL;
    }

    matrix *m_return = m_create(m1->Nrows, m1->Ncols, NULL);

    for (int i = 0; i < m_return->Nrows; i++)
    {
        for (int j = 0; j < m_return->Ncols; j++)
        {
            m_return->values[i][j] = m1->values[i][j] - m2->values[i][j];
        }
    }

    return m_return;
}

matrix *m_mult(matrix *m1, matrix *m2)
{
    if (m1->Ncols != m2->Nrows)
    {
        printf("Not equal");
        return NULL;
    }

    matrix *m_return = m_create(m1->Nrows, m2->Ncols, NULL);

    for (int i = 0; i < m1->Nrows; i++)
    {
        double *row = m_get_row(m1, i);

        for (int j = 0; j < m2->Ncols; j++)
        {
            double *col = m_get_col(m2, j);

            m_return->values[i][j] = m_dot(row, col, m1->Ncols);

            free(col);
        }

        free(row);
    }

    return m_return;
}

matrix *m_div(matrix *m1, matrix *m2)
{
    if (m2->Nrows != m2->Ncols)
    {
        return NULL;
    }

    return m_mult(m1, m_inverse(m2));
}
