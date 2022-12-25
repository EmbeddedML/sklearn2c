#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "Matrix.h"

Matrix *create_matrix(const int n_rows, const int n_cols)
{
    struct Matrix *m = malloc(sizeof(struct Matrix));
    m->rows = n_rows;
    m->cols = n_cols;
    m->data = (float **)malloc(sizeof(float *) * n_rows);
    for (int x = 0; x < n_rows; x++)
    {
        m->data[x] = (float *)calloc(n_cols, sizeof(float));
    }
    return m;
}

Matrix *copy_matrix(float *data, int n_rows, int n_cols)
{
    struct Matrix *matrix = create_matrix(n_rows, n_cols);
    int k = 0;
    for (int x = 0; x < n_rows; x++)
    {
        for (int y = 0; y < n_cols; y++)
        {
            matrix->data[x][y] = data[k];
            k++;
        }
    }
    return matrix;
}

Matrix *multiply(Matrix *mat1, Matrix *mat2)
{
    const int r1 = mat1->rows;
    const int c1 = mat1->cols;
    const int r2 = mat2->rows;
    const int c2 = mat2->cols;
    assert(c1 == r2);

    Matrix *mul = create_matrix(r1, c2);

    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            for (int k = 0; k < c1; k++)
            {
                mul->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
            }
        }
    }
    return mul;
}

Matrix *subtract(Matrix *mat1, Matrix *mat2)
{
    assert(mat1->cols == mat2->cols);
    assert(mat1->rows == mat2->rows);
    const int r = mat1->rows;
    const int c = mat1->cols;
    Matrix *sub = create_matrix(r, c);
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            sub->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
        }
    }

    return sub;
}

Matrix *transpose(Matrix *mat)
{
    Matrix *tran_mat = create_matrix(mat->cols, mat->rows);
    const int r = mat->rows;
    const int c = mat->cols;
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            tran_mat->data[j][i] = mat->data[i][j];
        }
    }

    return tran_mat;
}

//void print_matrix(Matrix *m)
//{
//    for (int x = 0; x < m->rows; x++)
//    {
//        for (int y = 0; y < m->cols; y++)
//        {
//            printf("%f ", m->data[x][y]);
//        }
//        printf("\n");
//    }
//    printf("\n");
//}
