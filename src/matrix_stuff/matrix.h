typedef unsigned int Index;

typedef struct {
    Index Nrows;
    Index Ncols;
    double **values;
} matrix;


char* m_to_string(matrix *m);

double m_det(matrix *m);

double m_dot(double *row, double *col, int size); 

double m_find_pivot(double *col);

double* m_get_row(matrix *m, Index i);

double* m_get_col(matrix *m, Index j);

double* r_sub(double *row1, double *row2, int size);

matrix* m_create(Index rows, Index cols, double *data[]);

matrix* m_scale(matrix *m, double scale);

matrix* m_transpose(matrix *m);

matrix* m_cofactor(matrix *m);

matrix* m_get_sub(matrix *m, Index row, Index col);

matrix* m_adj(matrix *m);

matrix* m_inverse(matrix *m);

matrix* m_add(matrix *m1, matrix *m2);

matrix* m_sub(matrix *m1, matrix *m2);

matrix* m_mult(matrix *m1, matrix *m2);

matrix* m_div(matrix *m1, matrix *m2);