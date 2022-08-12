#include <stdio.h>
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include <math.h>
#define MKL_INT size_t
#include "mkl.h"

void print_mat_info(mat *M, int a)
{
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            printf("%.4f  ", M->d[j * M->nrows + i]);
        }
        printf("\n");
    }
}

void matlab_debug(mat *M, char *a)
{
    FILE *fp = fopen(a, "w");
    printf("saving %s:(%d,%d)\n", a, M->nrows, M->ncols);
    setbuf(fp, NULL);
    print_mat_info(M, 10);
    for (int i = 0; i < M->nrows; ++i)
    {
        for (int j = 0; j < M->ncols; ++j)
        {
            double ans = matrix_get_element(M, i, j);
            fprintf(fp, "%lf,", ans);
            fflush(fp);
        }
        fprintf(fp, "\n");
        fflush(fp);
    }
    fclose(fp);
    printf("matrix saved\n");
}

/* C = beta*C + alpha*A(1:Anrows, 1:Ancols)[T]*B(1:Bnrows, 1:Bncols)[T] */
void submatrix_submatrix_mult_with_ab(mat *A, mat *B, mat *C,
                                      int Anrows, int Ancols, int Bnrows, int Bncols, int transa, int transb, double alpha, double beta)
{

    int opAnrows, opAncols, opBnrows, opBncols;
    if (transa == CblasTrans)
    {
        opAnrows = Ancols;
        opAncols = Anrows;
    }
    else
    {
        opAnrows = Anrows;
        opAncols = Ancols;
    }

    if (transb == CblasTrans)
    {
        opBnrows = Bncols;
        opBncols = Bnrows;
    }
    else
    {
        opBnrows = Bnrows;
        opBncols = Bncols;
    }

    if (opAncols != opBnrows)
    {
        printf("error in submatrix_submatrix_mult()");
        exit(0);
    }

    cblas_dgemm(CblasColMajor, transa, transb,
                opAnrows, opBncols,    // m, n,
                opAncols,              // k
                alpha, A->d, A->nrows, // 1, A, rows of A as declared in memory
                B->d, B->nrows,        // B, rows of B as declared in memory
                beta, C->d, C->nrows   // 0, C, rows of C as declared.
    );
}

void submatrix_submatrix_mult(mat *A, mat *B, mat *C, int Anrows, int Ancols, int Bnrows, int Bncols, int transa, int transb)
{
    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    submatrix_submatrix_mult_with_ab(A, B, C, Anrows, Ancols, Bnrows, Bncols, transa, transb, alpha, beta);
}

/* D = M(:,inds)' */
void matrix_get_selected_columns_and_transpose(mat *M, int *inds, mat *Mc)
{
    long long i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->nrows); i++)
        {
            col_vec = vector_new(M->nrows);
            matrix_get_col(M, inds[i], col_vec);
            matrix_set_row(Mc, i, col_vec);
            vector_delete(col_vec);
        }
    }
}

void matrix_set_selected_rows_with_transposed(mat *M, int *inds, mat *Mc)
{
    long long i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->ncols); i++)
        {
            col_vec = vector_new(Mc->nrows);
            matrix_get_col(Mc, i, col_vec);
            matrix_set_row(M, inds[i], col_vec);
            vector_delete(col_vec);
        }
    }
}

void linear_solve_Uxb(mat *A, mat *b)
{
    printf("Linear Solver retcode:%d\n", LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', //
                                                        b->nrows,
                                                        b->ncols,
                                                        A->d,
                                                        b->nrows, //ncols
                                                        b->d,
                                                        b->nrows));
}

void linear_solve_UTxb(mat *A, mat *b)
{
    LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', //
                   b->nrows,
                   b->ncols,
                   A->d,
                   b->nrows, //ncols
                   b->d,
                   b->nrows);
}

mat_coo *coo_matrix_new(int nrows, int ncols, long long capacity)
{
    mat_coo *M = (mat_coo *)malloc(sizeof(mat_coo));
    M->values = (double *)calloc(capacity, sizeof(double));
    M->rows = (int *)calloc(capacity, sizeof(int));
    M->cols = (int *)calloc(capacity, sizeof(int));
    M->nnz = 0;
    M->nrows = nrows;
    M->ncols = ncols;
    M->capacity = capacity;
    return M;
}

void coo_matrix_delete(mat_coo *M)
{
    free(M->values);
    free(M->cols);
    free(M->rows);
    free(M);
}

void coo_matrix_print(mat_coo *M)
{
    long long i;
    for (i = 0; i < M->nnz; i++)
    {
        printf("(%d, %d: %f), ", *(M->rows + i), *(M->cols + i), *(M->values + i));
    }
    printf("\n");
}

// 0-based interface
void set_coo_matrix_element(mat_coo *M, int row, int col, double val, int force_new)
{
    if (!(row >= 0 && row < M->nrows && col >= 0 && col < M->ncols))
    {
        printf("error: wrong index\n");
        exit(0);
    }
    if (!force_new)
    {
        int i;
        for (i = 0; i < M->nnz; i++)
        {
            if (*(M->rows + i) == row + 1 && *(M->cols + i) == col + 1)
            {
                *(M->values + i) = val;
                return;
            }
        }
    }
    if (M->nnz < M->capacity)
    {
        *(M->rows + M->nnz) = row + 1;
        *(M->cols + M->nnz) = col + 1;
        *(M->values + M->nnz) = val;
        M->nnz = M->nnz + 1;
        return;
    }
    printf("error: capacity exceeded. capacity=%d, nnz=%d\n", M->capacity, M->nnz);
    exit(0);
}

void coo_matrix_matrix_mult(mat_coo *A, mat *B, mat *C)
{
    /* 
    void mkl_dcoomm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *rowind , const MKL_INT *colind , 
        const MKL_INT *nnz , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    double alpha = 1.0, beta = 0.0;
    const char *trans = "N";
    const char *metadescra = "GXXF";
    mkl_dcoomm(
        trans, &(A->nrows), &(C->ncols),
        &(A->ncols), &(alpha), metadescra,
        A->values, A->rows, A->cols,
        &(A->nnz), B->d, &(B->nrows),
        &(beta), C->d, &(C->nrows));
}

void coo_matrix_transpose_matrix_mult(mat_coo *A, mat *B, mat *C)
{
    /* 
    void mkl_dcoomm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *rowind , const MKL_INT *colind , 
        const MKL_INT *nnz , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    double alpha = 1.0, beta = 0.0;
    const char *trans = "T";
    const char *metadescra = "GXXF";
    mkl_dcoomm(
        trans, &(A->nrows), &(C->ncols),
        &(A->ncols), &(alpha), metadescra,
        A->values, A->rows, A->cols,
        &(A->nnz), B->d, &(B->nrows),
        &(beta), C->d, &(C->nrows));
}

void coo_matrix_copy_to_dense(mat_coo *A, mat *B)
{
    int m = B->nrows, n = B->ncols;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix_set_element(B, i, j, 0.0);
        }
    }
    for (long long i = 0; i < A->nnz; i++)
    {
        matrix_set_element(B, *(A->rows + i) - 1, *(A->cols + i) - 1, *(A->values + i));
    }
}

double get_rand_uniform(VSLStreamStatePtr stream)
{
    double ans;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &ans, 0.0, 1.0);
    return ans;
}

double get_rand_normal(VSLStreamStatePtr stream)
{
    double ans;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &ans, 0.0, 1.0);
    return ans;
}

void gen_rand_coo_matrix(mat_coo *M, double density)
{
    VSLStreamStatePtr stream_u;
    VSLStreamStatePtr stream_n;
    // vslNewStream( &stream_u, BRNG, time(NULL));
    // vslNewStream( &stream_n, BRNG, time(NULL));
    vslNewStream(&stream_u, BRNG, 123);
    vslNewStream(&stream_n, BRNG, 456);
    int i, j;
    for (i = 0; i < M->nrows; i++)
    {
        for (j = 0; j < M->ncols; j++)
        {
            if (get_rand_uniform(stream_u) < density)
            {
                set_coo_matrix_element(M, i, j, get_rand_normal(stream_n), 1);
            }
        }
    }
}

void coo_matrix_sort_element(mat_coo *A)
{
    long long i, j;
    // seletion sort
    for (i = 0; i < A->nnz; i++)
    {
        for (j = i + 1; j < A->nnz; j++)
        {
            if ((A->rows[i] > A->rows[j]) ||
                (A->rows[i] == A->rows[j] && A->cols[i] > A->cols[j]))
            {
                double dtemp;
                int itemp;
                itemp = A->rows[i];
                A->rows[i] = A->rows[j];
                A->rows[j] = itemp;
                itemp = A->cols[i];
                A->cols[i] = A->cols[j];
                A->cols[j] = itemp;
                dtemp = A->values[i];
                A->values[i] = A->values[j];
                A->values[j] = dtemp;
            }
        }
    }
}

void csr_matrix_delete(mat_csr *M)
{
    free(M->values);
    free(M->cols);
    free(M->pointerB);
    free(M->pointerE);
    free(M);
}

void csr_matrix_print(mat_csr *M)
{
    long long i;
    printf("values: ");
    for (i = 0; i < M->nnz; i++)
    {
        printf("%f ", M->values[i]);
    }
    printf("\ncolumns: ");
    for (i = 0; i < M->nnz; i++)
    {
        printf("%d ", M->cols[i]);
    }
    printf("\npointerB: ");
    for (i = 0; i < M->nrows; i++)
    {
        printf("%d\t", M->pointerB[i]);
    }
    printf("\npointerE: ");
    for (i = 0; i < M->nrows; i++)
    {
        printf("%d\t", M->pointerE[i]);
    }
    printf("\n");
}

mat_csr *csr_matrix_new()
{
    mat_csr *M = (mat_csr *)malloc(sizeof(mat_csr));
    return M;
}

void csr_init_from_coo(mat_csr *D, mat_coo *M)
{
    D->nrows = M->nrows;
    D->ncols = M->ncols;
    D->pointerB = (int *)malloc(D->nrows * sizeof(int));
    D->pointerE = (int *)malloc(D->nrows * sizeof(int));
    D->cols = (int *)calloc(M->nnz, sizeof(int));
    D->nnz = M->nnz;

    // coo_matrix_sort_element(M);
    D->values = (double *)malloc(M->nnz * sizeof(double));
    memcpy(D->values, M->values, M->nnz * sizeof(double));

    int current_row, cursor = 0;
    for (current_row = 0; current_row < D->nrows; current_row++)
    {
        D->pointerB[current_row] = cursor + 1;
        while (M->rows[cursor] - 1 == current_row)
        {
            D->cols[cursor] = M->cols[cursor];
            cursor++;
        }
        D->pointerE[current_row] = cursor + 1;
    }
}

void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C)
{
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char *transa = "N";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
               &(A->ncols), &alpha, matdescra,
               A->values, A->cols, A->pointerB,
               A->pointerE, B->d, &(B->nrows),
               &beta, C->d, &(C->nrows));
}

void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C)
{
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char *transa = "T";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
               &(A->ncols), &alpha, matdescra,
               A->values, A->cols, A->pointerB,
               A->pointerE, B->d, &(B->nrows),
               &beta, C->d, &(C->nrows));
}

/*********************************Yuyang Xie*****************************/

/*  k*n = k*k k*k n*k  */
void svd_row_cut(mat *A, mat *U, mat *E, mat *V)
{
    long long m = A->nrows;
    long long n = A->ncols;
    long long i, j;
    // mat *A_in = matrix_new(m,n);;

    // matrix_copy(A_in, A);
    // printf("dong tai sheng qing\n");
    // double *u = (double*)malloc(m*m*sizeof(double));
    mat *Vt = matrix_new(m, n);
    // printf("svd is running\n");
    // LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'S', m, n, A->d, n, E->d, U->d, m, vt, n, superb);

    // LAPACKE_dgesdd( int matrix_layout, char jobz, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt );
    LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', m, n, A->d, m, E->d, U->d, m, Vt->d, m);
    //printf("Complete Lapack svd\n\n");
    matrix_build_transpose(V, Vt);
    // printf("svd_row_cut is over\n");

    // matrix_delete(A_in);
    matrix_delete(Vt);
}

/* C = A*B & D = A^T*C */
void matrix_union_matrix_mult_disk_mem(FILE *A, mat *B, mat *C, mat *D, int row, int col, int row_size)
{
    int read_row_size = row_size;

    double alpha, beta, gama;
    long long i, j; // count
    long long m = row, n = col, k = B->ncols;
    //float *M_f = (float*)malloc(read_row_size*n*sizeof(float));
    double *M = (double *)malloc(read_row_size * n * sizeof(double));
    // double *g_row= (double*)malloc(k*sizeof(double)); //C's row vector'
    //printf("matrix_union_matrix_mult_disk_mem is running\n");

    alpha = 1.0;
    beta = 0.0;
    gama = 1.0;
    struct timeval start_timeval_1, end_timeval_1;
    struct timeval start_timeval_2, end_timeval_2;
    double sum = 0;
    double time_1, time_2;
    gettimeofday(&start_timeval_1, NULL); //time_1
    //  #pragma omp parallel shared(D) private(i)
    //     {
    //         #pragma omp parallel for
    //         for(i=0; i < (D->nrows*D->ncols); i++){
    //             D->d[i] = 0.0;
    //         }
    //     }

    for (i = 0; i < m; i += row_size)
    {
        if (row_size > (m - i))
            read_row_size = m - i;
        gettimeofday(&start_timeval_2, NULL); //time_2
        //fread(M_f, sizeof(float), n*read_row_size, A);
        /*#pragma omp parallel shared(M,M_f,n,read_row_size) private(j) 
        {
            #pragma omp parallel for
            for(j=0; j < n*read_row_size; j++){
                M[j] = M_f[j];          
            }
        }
        */
        fread(M, sizeof(double), n * read_row_size, A);
        gettimeofday(&end_timeval_2, NULL);
        sum += get_seconds_frac(start_timeval_2, end_timeval_2);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, n, B->d, B->ncols, beta, C->d + i * k, C->ncols);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, C->ncols, read_row_size, alpha, M, n, C->d + i * k, C->ncols, gama, D->d, D->ncols);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, read_row_size, B->d, n, beta, C->d+i*k, read_row_size);
        //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, C->ncols, read_row_size, alpha, M, read_row_size, C->d+i*k, read_row_size, gama, D->d, n);

        //printf("---debug----\n");
        /*
        double *C_tmp=(double *)malloc(read_row_size*k*sizeof(double));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, read_row_size, B->d, n, beta, C_tmp, read_row_size);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, C->ncols, read_row_size, alpha, M, read_row_size, C_tmp, read_row_size, gama, D->d, n);
        
        vec *c_row = vector_new(k);
        for (int pp=0;pp<read_row_size;pp++){
            int qq;
            #pragma omp parallel shared(c_row,C_tmp,pp,read_row_size,k) private(qq)
            {
            #pragma omp for
            for (qq=0;qq<k;qq++){
                matrix_set_element(C,i+pp,qq,C_tmp[qq*read_row_size+pp]);
            }
            }
        }
        vector_delete(c_row);
        free(C_tmp);
        */
    }

    gettimeofday(&end_timeval_1, NULL);
    time_1 = get_seconds_frac(start_timeval_1, end_timeval_1);

    time_2 = sum;

    printf("Time for reading data file_(fread-time): %g second\n", time_2);
    printf("Time for matrix_union_matrix_mult: %g second\n", time_1);
    //printf("matrix_union_mem is %d KB\n",  getCurrentRSS()/1024);

    //free(M_f);
    free(M);
}

mat_csc *csc_matrix_new(int nrows, int ncols, long long capacity)
{
    mat_csc *M = (mat_csc *)malloc(sizeof(mat_csc));
    M->pointerB = (int *)malloc((nrows + 1) * sizeof(int));
    M->pointerE = (int *)malloc((nrows + 1) * sizeof(int));
    M->values = (double *)calloc(capacity, sizeof(double));
    M->rows = (int *)calloc(capacity, sizeof(int));
    M->nnz = capacity;
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}

void csr2csc(mat_csr *A, mat_csc *B)
{
    int job[6];
    job[0] = 0;
    job[1] = 1;
    job[2] = 1;
    job[5] = 1;
    int info;
    //void mkl_dcsrcsc (const MKL_INT *job , const MKL_INT *n , double *acsr , MKL_INT *ja ,MKL_INT *ia , double *acsc , MKL_INT *ja1 , MKL_INT *ia1 , MKL_INT *info );
    mkl_dcsrcsc(job, &A->nrows, A->values, A->cols, A->pointerB, B->values, B->rows, B->pointerB, &info);
}

void csc_matrix_delete(mat_csc *M)
{
    free(M->values);
    free(M->rows);
    free(M->pointerB);
    free(M->pointerE);
    free(M);
}

//mat M is a size(n,1) vector
void csr_init_from_diag(mat_csr *A, mat *M)
{
    A->nrows = M->nrows;
    A->ncols = M->nrows;
    A->pointerB = (int *)malloc(A->nrows * sizeof(int));
    A->pointerE = (int *)malloc(A->nrows * sizeof(int));
    A->cols = (int *)calloc(M->nrows, sizeof(int));
    A->nnz = M->nrows;

    // coo_matrix_sort_element(M);
    A->values = (double *)malloc(M->nrows * sizeof(double));
    memcpy(A->values, M->d, M->nrows * sizeof(double));

    int current_row, cursor = 0;
    for (current_row = 0; current_row < A->nrows; current_row++)
    {
        A->cols[current_row] = current_row + 1;
        A->pointerB[current_row] = cursor + 1;
        cursor++;
        A->pointerE[current_row] = cursor + 1;
    }
}

//void csr_compute_eig(mat_csr *A)
//{
//void dfeast_scsrev (const char * uplo, const MKL_INT * n, const double * a, const
//MKL_INT * ia, const MKL_INT * ja, MKL_INT * fpm, double * epsout, MKL_INT * loop,
//const double * emin, const double * emax, MKL_INT * m0, double * e, double * x,
//MKL_INT * m, double * res, MKL_INT * info);
//
//}

/*********************************Yuyang Xie*****************************/
