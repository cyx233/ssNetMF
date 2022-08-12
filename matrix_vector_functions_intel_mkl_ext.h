#pragma once

#include "matrix_vector_functions_intel_mkl.h"

/* 
    The rows/cols fields use 1-based indexing. This is due to an undocumented 
    feature of MKL library: if you are trying to cooperate sparse matrix with 
    row-major layout dense matrices, then the interfaces assume the indexing 
    is 0-based. Otherwise if you use column major dense matrices, you should 
    use 1-based indexing.
*/
typedef struct
{
    int nrows, ncols;
    long long nnz;      // number of non-zero element in the matrix.
    long long capacity; // number of possible nnzs.
    double *values;
    int *rows, *cols;
} mat_coo;

typedef struct
{
    long long nnz;
    int nrows, ncols;
    double *values;
    int *cols;
    int *pointerB, *pointerE;
} mat_csr;

typedef struct
{
    long long nnz;
    int nrows, ncols;
    double *values;
    int *rows;
    int *pointerB, *pointerE;
} mat_csc;

void linear_solve_UTxb(mat *A, mat *b);
void linear_solve_UxX(mat *A, mat *b);

// initialize with sizes, the only interface that allocates space for coo struct
mat_coo *coo_matrix_new(int nrows, int ncols, long long capacity);

// collect allocated space.
void coo_matrix_delete(mat_coo *M);

void coo_matrix_print(mat_coo *M);

void set_coo_matrix_element(mat_coo *M, int row, int col, double val, int force_new);

void coo_matrix_matrix_mult(mat_coo *A, mat *B, mat *C);

void coo_matrix_transpose_matrix_mult(mat_coo *A, mat *B, mat *C);

void coo_matrix_copy_to_dense(mat_coo *A, mat *B);

double get_rand_uniform(VSLStreamStatePtr stream);

double get_rand_normal(VSLStreamStatePtr stream);

void gen_rand_coo_matrix(mat_coo *M, double density);

void coo_matrix_sort_element(mat_coo *A);

// return a pointer, but nothing inside.
mat_csr *csr_matrix_new();

// collect the space
void csr_matrix_delete(mat_csr *M);

void csr_matrix_print(mat_csr *M);

// the only interface that allocates space for mat_csr struct and initialize with M
void csr_init_from_coo(mat_csr *D, mat_coo *M);

void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C);

void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C);

//Algorithms by Xu Feng
//void LUfraction(mat *A, mat *L);

//void eigSVD(mat *A, mat *U, mat *S, mat *V);

//void basic_rPCA(mat_csr *M, mat *U, mat *S, mat *V,int k,int p);

//Algorithms by Yuyang Xie

void svd_row_cut(mat *A, mat *U, mat *E, mat *V);

void matrix_union_matrix_mult_disk_mem(FILE *A, mat *B, mat *C, mat *D, int row, int col, int row_size);

void csr2csc(mat_csr *A, mat_csc *B);

mat_csc *csc_matrix_new(int nrows, int ncols, long long capacity);

void csc_matrix_delete(mat_csc *M);

void csr_init_from_diag(mat_csr *A, mat *M);

//void rsymsvds(mat_csr *A, mat *U, mat *S, mat *V, int k, int p);

//void frsymembeddings(mat_csr *A, mat *U, mat *S, int k, int p);

//void rsymSVD_SP(mat_csr *A, int m, int n, int k, int q, mat *U, mat *S, mat *V);

void print_mat_info(mat *M, int a);

void matlab_debug(mat *M, char *a);