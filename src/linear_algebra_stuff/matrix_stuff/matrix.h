#ifndef MATRIX_H
#define MATRIX_H

typedef unsigned int Index;

typedef struct {
    Index Nrows;
    Index Ncols;
    double **values;
} matrix;

// Takes a matrix *m and returns a string containing the values of the matrix.
char* m_to_string(matrix *m);

// Takes a pointer to an array and it's size and returns a string containing the values of the array.
char* r_to_string(double *row, int size);

// Takes a matrix *m and returns the determinant of it.
double m_det(matrix *m);

// Takes 2 pointers arrays and their size and return the dot product of them.
double m_dot(double *row, double *col, int size);

// Takes a matrix *m and an index i and returns the row at that index.
double* m_get_row(matrix *m, Index i);

// Takes a matrix *m  and an index j and returns the column at that jndex.
double* m_get_col(matrix *m, Index j);

// Takes a num rows, cols and a pointer a 2D array data and returns a matrix the size of the rows and cols containing the data.
matrix* m_create(Index rows, Index cols, double **data);

// Takes a matrix *m and a scale and returns a scaled matrix.
matrix* m_scale(matrix *m, double scale);

// Takes a matrix *m and returns the transpose of it.
matrix* m_transpose(matrix *m);

// Takes a matrix *m and returns the cofactor matrix of it.
matrix* m_cofactor(matrix *m);

// Takes a matrix *m and the Index's row and col and returns the submatrix of those Indecies.
matrix* m_get_sub(matrix *m, Index row, Index col);

// Takes a matrix *m and returns the Adjoint matrix of it.
matrix* m_adj(matrix *m);

// Takes a matrix *m and returns the inverse of it.
matrix* m_inverse(matrix *m);

// Takes 2 matrices *m1 and *m2 and returns the added matrix.
matrix* m_add(matrix *m1, matrix *m2);

// Takes 2 matrices *m1 and *m2 and returns the subtracted matrix.
matrix* m_sub(matrix *m1, matrix *m2);

// Takes 2 matrices *m1 and *m2 and returns the multiplied matrix.
matrix* m_mult(matrix *m1, matrix *m2);
/*
    CHANGE THIS FUNCTION TO A BETTER FASTER MORE OPTIMIZED FUNCTION.
*/

// Takes 2 matrices *m1 and *m2 and returns the "divided" matrix.
matrix* m_div(matrix *m1, matrix *m2);

#endif // MATRIX_H