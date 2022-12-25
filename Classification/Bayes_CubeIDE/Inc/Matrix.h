#ifndef MATRIX_H
#define MATRIX_H

struct Matrix
{
    int rows;     // number of rows
    int cols;     // number of columns
    float **data; // a pointer to an array of n_rows pointers to rows
};

typedef struct Matrix Matrix;

Matrix *create_matrix(const int n_rows, const int n_cols);
Matrix *copy_matrix(float *data, int n_rows, int n_cols);
Matrix *multiply(Matrix *mat1, Matrix *mat2);
Matrix *subtract(Matrix *mat1, Matrix *mat2);
Matrix *transpose(Matrix *mat);
void print_matrix(Matrix *m);

#endif
