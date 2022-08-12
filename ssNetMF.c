#include <stdio.h>
#include <stdlib.h>
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include "freigs.h"
#include <omp.h>
#include <float.h>
#include <string.h>
/**
 * @brief ARPACK header for dsaupd_.
 */

char filename[200] = "ppi_coo.txt";
char outputname[200] = "ppi";
int n = 3890;
long long nnz = 76584;

BOOL outputflag = true;
int window_size = 10;
int b = 1;
int h = 256;
int dim = 128;
int batch = 128;
int block = 32;
int q = 5;
int s1 = 100;
int s2 = 100;
int s3 = 100;
int use_freigs_convex = 0;
double alpha = 0.375;

// get the D^{-1/2}
void caculate_laplacian(mat_csr *A, mat *d_rt, int n)
{
    int i, j;
#pragma omp parallel shared(A, d_rt, n) private(i, j)
    {
#pragma omp for
        for (i = 0; i < n; i++)
        {
            d_rt->d[i] = 0;
            for (j = A->pointerB[i] - 1; j < A->pointerE[i] - 1; j++)
            {
                d_rt->d[i] += A->values[j];
            }
            d_rt->d[i] = 1.0 / sqrt(d_rt->d[i]);
        }
    }
}

void approximate_normalized_graph_laplacian(mat_csr *A, mat *D_rt_invU, mat *evals, mat *d_rt)
{
    int n = A->ncols;
    int i, j;
    mat *d_rta_b = matrix_new(n, 1);
    caculate_laplacian(A, d_rt, n); //D^{-1/2}
    matrix_copy(d_rta_b, d_rt);
    printf("We choose alpha:%.8f\n", alpha);
#pragma omp parallel shared(d_rta_b) private(i)
    {
#pragma omp for
        for (i = 0; i < n; i++)
        {
            d_rta_b->d[i] = pow(d_rta_b->d[i], alpha * 2); //D^{-alpha}
        }
    }
//computing D^{-alpha} A D^{-alpha}
#pragma omp parallel shared(A, d_rta_b, n) private(i, j)
    {
#pragma omp for
        for (i = 0; i < n; i++)
        {
            for (j = A->pointerB[i] - 1; j < A->pointerE[i] - 1; j++)
            {
                A->values[j] = A->values[j] * matrix_get_element(d_rta_b, i, 0);
                A->values[j] = A->values[j] * matrix_get_element(d_rta_b, A->cols[j] - 1, 0);
            }
        }
    }
    mat *U = matrix_new(n, h);
    struct timeval start1, end1, end2;
    gettimeofday(&start1, NULL);
    if (use_freigs_convex)
        freigs_convex(A, U, evals, h, q, s1);
    else
        eigs(A, U, evals, h, 300, s1);

    double maxs = -10000000;
    double mins = DBL_MAX;
    for (i = 0; i < h; i++)
    {
        maxs = max(maxs, evals->d[i]);
        mins = min(mins, evals->d[i]);
    }
    printf("After randomized Eig, evals max:%f---evals min%f\n", maxs, mins);
    printf("Computing D^{-1/x}U...\n");
    mat *d_rt1_x = matrix_new(n, 1);
    matrix_copy(d_rt1_x, d_rt);
#pragma omp parallel shared(d_rt1_x) private(i)
    {
#pragma omp for
        for (i = 0; i < n; i++)
        {
            d_rt1_x->d[i] = pow(d_rt1_x->d[i], 1 - alpha * 2); //D^{-1/2+alpha}
        }
    }
    mat_csr *D_rt_inv = csr_matrix_new();
    csr_init_from_diag(D_rt_inv, d_rt1_x);
    csr_matrix_matrix_mult(D_rt_inv, U, D_rt_invU);
    matrix_delete(U);
    csr_matrix_delete(D_rt_inv);
    matrix_delete(d_rta_b);
    matrix_delete(d_rt1_x);
    gettimeofday(&end1, NULL);
    printf("the total time of randomized eigen decomposition %f seconds\n", get_seconds_frac(start1, end1));
    return;
}

void deepwalk_filter(mat *evals, int h)
{
    int i;
    for (i = 0; i < h; i++)
    {
        double x = evals->d[i];
        if (x >= 1)
            evals->d[i] = 1;
        else
            evals->d[i] = x * (1 - pow(x, window_size)) / (1 - x) / window_size;
        evals->d[i] = max(evals->d[i], 0);
    }
    return;
}

void deepwalk_appro(mat *evals, mat *D_rt_invU, mat *Mdiag)
{
    int i;
    printf("Begin deepwalk_appro\n");
    mat *Mtmp = matrix_new(h, h);
    initialize_identity_matrix(Mdiag);
    initialize_identity_matrix(Mtmp);
    mat *Msquare = matrix_new(h, h);
    mat *D_rt_invUt = matrix_new(h, n);
    matrix_build_transpose(D_rt_invUt, D_rt_invU);
    matrix_matrix_mult(D_rt_invUt, D_rt_invU, Msquare);
    matrix_delete(D_rt_invUt);
    mat *diag = matrix_new(h, h);
#pragma omp parallel shared(diag) private(i)
    {
#pragma omp parallel for
        for (i = 0; i < (diag->nrows); i++)
        {
            matrix_set_element(diag, i, i, evals->d[i]);
        }
    }
    mat *M = matrix_new(h, h);
    matrix_matrix_mult(Msquare, diag, M);
    matrix_delete(diag);
    matrix_delete(Msquare);
    printf("begin window iter\n");
    mat *M_window_iter = matrix_new(h, h);
    for (i = 1; i < window_size; i++)
    {
        matrix_matrix_mult(Mtmp, M, M_window_iter);
        matrix_copy(Mtmp, M_window_iter);
        matrix_matrix_add(M_window_iter, Mdiag);
    }
    matrix_delete(M_window_iter);
    matrix_delete(M);
    matrix_delete(Mtmp);
}

typedef struct
{
    double eval;
    int e_ind;
} node;

int cmp(node *a, node *b)
{
    return a->eval < b->eval ? 1 : -1;
}

int spSignGen(int nrow, int ncol, int gamma, mat_csc *spMat, int *unique_pos)
{
    char *count = calloc((nrow + 7) / 8, sizeof(char));
    spMat->nnz = gamma * ncol;
#pragma omp parallel shared(spMat, gamma, ncol, nrow, count)
    {
#pragma omp for
        for (int i = 0; i < ncol; ++i)
        {
            spMat->pointerB[i] = i * gamma + 1;
            spMat->pointerE[i] = (i + 1) * gamma + 1;
            char *bitmap = calloc((nrow + 7) / 8, sizeof(char));
            int *key = calloc(gamma, sizeof(int));
            int *value = calloc(gamma, sizeof(int));
            int *ans = calloc(gamma, sizeof(int));
            int k, j;
            for (j = 0; j < gamma; ++j)
            {
                int dst = rand() % (nrow - i) + i;
                if (dst < gamma)
                {
                    ans[dst] = ans[j];
                    ans[j] = dst;
                }
                else if (bitmap[dst / 8] & (1 << (dst % 8)))
                {
                    for (k = 0; k < gamma; ++k)
                        if (key[k] == dst)
                        {
                            dst = value[k];
                            value[k] = ans[j];
                            ans[j] = dst;
                            break;
                        }
                }
                else
                {
                    for (k = 0; k < gamma; ++k)
                        if (key[k] == 0)
                        {
                            key[k] = dst;
                            value[k] = ans[j];
                            ans[j] = dst;
                            bitmap[dst / 8] |= 1 << (dst % 8);
                            break;
                        }
                }
            }
            for (j = 0; j < gamma; ++j)
            {
                spMat->rows[i * gamma + j] = ans[j];
                spMat->values[i * gamma + j] = 2 * (rand() % 2) - 1;
                count[ans[j] / 8] |= 1 << (ans[j] % 8);
            }
            free(bitmap);
            free(key);
            free(value);
            free(ans);
        }
    }
    int unique_num = 0;
    for (int i = 0; i < nrow; ++i)
    {
        if (count[i / 8] & (1 << (i % 8)))
        {
            unique_pos[unique_num] = i;
            unique_num += 1;
        }
    }
    free(count);
    return unique_num;
}

void approximate_deepwalk_matrix_using_single_pass_sketch_svd(mat *evals, mat *D_rt_invU, mat *d_rt, double para, mat *U, mat *S, mat *V)
{
    int i, j;
    struct timeval start1_5, end1_5;
    gettimeofday(&start1_5, NULL);
    printf("s2 = %d, s3 = %d\n", s2, s3);

    mat *M = matrix_new(h, n);
    mat_csr *d_rt_csr = csr_matrix_new();
    csr_init_from_diag(d_rt_csr, d_rt);
    mat *F = matrix_new(n, h);

    csr_matrix_matrix_mult(d_rt_csr, D_rt_invU, F); //d^{-1+a}U
    csr_matrix_delete(d_rt_csr);

    if (alpha == 0.5)
    {
        deepwalk_filter(evals, h);
        printf("para:%f\n", para);
        para_matrix_mult(evals, para);

        mat_csr *evals_csr = csr_matrix_new();
        csr_init_from_diag(evals_csr, evals);
        mat *F_T = matrix_new(h, n);
        matrix_build_transpose(F_T, F);
        csr_matrix_matrix_mult(evals_csr, F_T, M);

        matrix_delete(F_T);
        csr_matrix_delete(evals_csr);
    }
    else
    {
        mat_csr *evals_csr = csr_matrix_new();
        csr_init_from_diag(evals_csr, evals);

        mat *temp = matrix_new(h, h);
        matrix_transpose_matrix_mult(D_rt_invU, D_rt_invU, temp); //U^Td^{-1+2a}U

        mat *K_T = matrix_new(h, h);
        csr_matrix_matrix_mult(evals_csr, temp, K_T); //evals*U^Td^{-1+2a}U

        matrix_delete(temp);

        mat *K = matrix_new(h, h);
        matrix_build_transpose(K, K_T);

        matrix_delete(K_T);

        vec *ones = vector_new(h);
        for (int i = 0; i < h; ++i)
            ones->d[i] = 1;
        mat *Ki = matrix_new(h, h);
        mat *Kiter = matrix_new(h, h);
        mat *evals_new = matrix_new(h, h);
        initialize_diagonal_matrix(Ki, ones);
        initialize_diagonal_matrix(evals_new, ones);

        vector_delete(ones);

        for (int i = 0; i < window_size - 1; ++i)
        {
            matrix_matrix_mult(K, Ki, Kiter);
            matrix_copy(Ki, Kiter);
            matrix_matrix_add(Kiter, evals_new);
        }
        mat *evals_new_ = matrix_new(h, h);
        csr_matrix_matrix_mult(evals_csr, evals_new, evals_new_);
        para_matrix_mult(evals_new_, para / window_size);

        csr_matrix_delete(evals_csr);
        matrix_delete(evals_new);
        matrix_delete(Ki);
        matrix_delete(Kiter);

        mat *F_T = matrix_new(h, n);
        matrix_build_transpose(F_T, F);
        matrix_matrix_mult(evals_new_, F_T, M);

        matrix_delete(evals_new_);
        matrix_delete(F_T);
    }

    gettimeofday(&end1_5, NULL);
    printf("the total time of deepwalk_appro cost %f seconds\n", get_seconds_frac(start1_5, end1_5));
    printf("begin spSignGen1\n");
    int gamma = 8;
    int col_num = dim + s2;

    mat_csc *spMat1 = csc_matrix_new(n, col_num, gamma * col_num);
    int *unique_pos = calloc(col_num * gamma, sizeof(int));
    int unique_num = spSignGen(n, col_num, 8, spMat1, unique_pos);
    int *num2index = calloc(n, sizeof(int));
    for (int i = 0; i < unique_num; ++i)
        num2index[unique_pos[i]] = i;

    mat *M2 = matrix_new(n, unique_num);
    printf("calculate M3=F*M2\n");
#pragma omp parallel shared(unique_num, M, D_rt_invU, M2)
    {
#pragma omp for
        for (int i = 0; i < unique_num; ++i)
        {
            vec *col_vec = vector_new(h);
            vec *ans_vec = vector_new(n);
            matrix_get_col(M, unique_pos[i], col_vec);
            matrix_vector_mult(F, col_vec, ans_vec);
            matrix_set_col(M2, i, ans_vec);
            vector_delete(col_vec);
            vector_delete(ans_vec);
        }
    }

    printf("calculate Y=M3*spMat1\n");
    mat *Y = matrix_new(n, col_num);
#pragma omp parallel shared(col_num, M2, spMat1, Y)
    {
#pragma omp for
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < col_num; ++j)
            {
                double sum = 0;
                for (int k = spMat1->pointerB[j] - 1; k < spMat1->pointerE[j] - 1; ++k)
                {
                    double ans = matrix_get_element(M2, i, num2index[spMat1->rows[k]]);
                    if (ans < 1)
                        ans = 0;
                    else
                        ans = log(ans);
                    if (spMat1->values[k] > 0)
                        sum += ans;
                    else
                        sum -= ans;
                }
                matrix_set_element(Y, i, j, sum);
            }
        }
    }

    csc_matrix_delete(spMat1);
    free(unique_pos);
    free(num2index);

    printf("begin spSignGen2\n");
    gamma = 8;
    int col_num2 = dim + s3;
    mat_csc *spMat2 = csc_matrix_new(n, col_num2, gamma * col_num2);
    unique_pos = calloc(col_num2 * gamma, sizeof(int));
    unique_num = spSignGen(n, col_num2, 8, spMat2, unique_pos);
    num2index = calloc(n, sizeof(int));
    for (int i = 0; i < unique_num; ++i)
        num2index[unique_pos[i]] = i;

    printf("calculate M3=F2*M2\n");
    mat *M3 = matrix_new(unique_num, unique_num);
#pragma omp parallel shared(unique_num, D_rt_invU, M, M3)
    {
#pragma omp for
        for (int i = 0; i < unique_num; ++i)
        {
            for (int j = 0; j < unique_num; ++j)
            {
                double sum = 0;
                for (int k = 0; k < h; ++k)
                    sum += matrix_get_element(F, unique_pos[i], k) * matrix_get_element(M, k, unique_pos[j]);
                if (sum > 1)
                    matrix_set_element(M3, i, j, log(sum));
            }
        }
    }

    printf("calculate M4=M3*spMat2\n");
    mat *M4 = matrix_new(unique_num, col_num2);
    free(unique_pos);

#pragma omp parallel shared(col_num2, unique_num, spMat2, M3, M4)
    {
#pragma omp for
        for (int i = 0; i < unique_num; ++i)
        {
            for (int j = 0; j < col_num2; ++j)
            {
                double sum = 0;
                for (int k = spMat2->pointerB[j] - 1; k < spMat2->pointerE[j] - 1; ++k)
                {
                    double ans = matrix_get_element(M3, i, num2index[spMat2->rows[k]]);
                    if (spMat2->values[k] > 0)
                        sum += ans;
                    else
                        sum -= ans;
                }
                matrix_set_element(M4, i, j, sum);
            }
        }
    }
    matrix_delete(M3);

    printf("calculate Z=spMat2'*M4\n");
    mat *Z = matrix_new(col_num2, col_num2);
#pragma omp parallel shared(col_num2, spMat2, Z)
    {
#pragma omp for
        for (int i = 0; i < col_num2; ++i)
        {
            for (int j = 0; j < col_num2; ++j)
            {
                double sum = 0;
                for (int k = spMat2->pointerB[i] - 1; k < spMat2->pointerE[i] - 1; ++k)
                {
                    double ans = matrix_get_element(M4, num2index[spMat2->rows[k]], j);
                    if (spMat2->values[k] > 0)
                        sum += ans;
                    else
                        sum -= ans;
                }
                matrix_set_element(Z, i, j, sum);
            }
        }
    }
    matrix_delete(M4);
    free(num2index);

    printf("Q,R=qr(Y)\n");
    QR_factorization_getQ_inplace(Y);

    printf("calculate Temp=spMat2'*Q\n");
    mat *Temp = matrix_new(col_num2, col_num);
#pragma omp parallel shared(col_num2, col_num, spMat2, Z)
    {
#pragma omp for
        for (int i = 0; i < col_num2; ++i)
        {
            for (int j = 0; j < col_num; ++j)
            {
                double sum = 0;
                for (int k = spMat2->pointerB[i] - 1; k < spMat2->pointerE[i] - 1; ++k)
                {
                    double ans = matrix_get_element(Y, spMat2->rows[k], j);
                    if (spMat2->values[k] > 0)
                        sum += ans;
                    else
                        sum -= ans;
                }
                matrix_set_element(Temp, i, j, sum);
            }
        }
    }

    mat *UU = matrix_new(col_num2, col_num);
    mat *TT = matrix_new(col_num, col_num);
    compact_QR_factorization(Temp, UU, TT);
    matrix_delete(Temp);

    mat *T = matrix_new(col_num, col_num2);
    matrix_transpose_matrix_mult(UU, Z, T); //UU'Z

    printf("calculate T = Temp\\Z\n");
    linear_solve_Uxb(TT, T); //T=TT\UU'Z

    printf("calculate C = (Temp\\T\')\'\n");
    mat *C = matrix_new(col_num, col_num);
    mat *C_T = matrix_new(col_num, col_num);
    matrix_matrix_mult(T, UU, C);   //TUU
    matrix_build_transpose(C_T, C); //UU'T'
    linear_solve_Uxb(TT, C_T);      //C'=TT\UU'T'
    matrix_build_transpose(C, C_T);

    matrix_delete(UU);
    matrix_delete(TT);
    matrix_delete(T);
    matrix_delete(C_T);

    printf("svd\n");
    mat *Uc = matrix_new(col_num, col_num);
    mat *Sc = matrix_new(col_num, col_num);
    mat *VcT = matrix_new(col_num, col_num);

    singular_value_decomposition(C, Uc, Sc, VcT);

    mat *Vc = matrix_new(col_num, col_num);
    matrix_build_transpose(Vc, VcT);
    matrix_delete(VcT);

    mat *Uc2 = matrix_new(col_num, dim);
    mat *Vc2 = matrix_new(col_num, dim);

    printf("cut\n");
    vec *col_vec = vector_new(col_num);
    for (int i = 0; i < dim; ++i)
    {
        matrix_get_col(Uc, i, col_vec);
        matrix_set_col(Uc2, i, col_vec);
        matrix_get_col(Vc, i, col_vec);
        matrix_set_col(Vc2, i, col_vec);
    }
    vector_delete(col_vec);

    for (int i = 0; i < dim; ++i)
        matrix_set_element(S, i, 0, matrix_get_element(Sc, i, i));

    printf("U=Q*Uc,V=Q*Vc\n");
    matrix_matrix_mult(Y, Uc2, U);
    matrix_matrix_mult(Y, Vc2, V);
    return;
}

void netmf(mat_csr *A, double para)
{
    int i, j;
    int n = A->ncols;
    printf("Running NetMF for a large window size...\n");
    printf("Window size is set to be %d\n", window_size);
    mat *D_rt_invU = matrix_new(n, h);
    mat *evals = matrix_new(h, 1);
    mat *d_rt = matrix_new(n, 1);
    //perform random eigen-decomposition of D^{-1/2} A D^{-1/2}
    //keep top h eigen values and vectors
    approximate_normalized_graph_laplacian(A, D_rt_invU, evals, d_rt);
    printf("Using single_pass_sketch algorithm\n");
    mat *U = matrix_new(n, dim);
    mat *S = matrix_new(dim, 1);
    mat *V = matrix_new(n, dim);
    approximate_deepwalk_matrix_using_single_pass_sketch_svd(evals, D_rt_invU, d_rt, para, U, S, V);
    matrix_delete(D_rt_invU);
    matrix_delete(evals);
    matrix_delete(d_rt);
    if (outputflag == true)
    {
        char tmp1[50], tmp2[50];
        strcpy(tmp1, outputname);
        strcpy(tmp2, outputname);
        writeMatrix2file(U, strcat(tmp1, ".U"));
        writeMatrix2file(S, strcat(outputname, ".S"));
        writeMatrix2file(V, strcat(tmp2, ".V"));
    }
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(V);
}

int main(int argc, char **argv)
{
    long long i;
    window_size = atoi(argv[2]);
    strcpy(filename, argv[4]);
    strcpy(outputname, argv[6]);
    n = atoi(argv[8]);
    nnz = atoll(argv[10]);
    b = atoi(argv[12]);
    h = atoi(argv[14]);
    dim = atoi(argv[16]);
    batch = atoi(argv[18]);
    q = atoi(argv[20]);
    s1 = atoi(argv[22]);
    s2 = atoi(argv[24]);
    s3 = atoi(argv[26]);
    use_freigs_convex = atoi(argv[28]);
    alpha = atof(argv[30]);

    printf("*************************************************************************************\n");
    printf("single-pass sketch SVD algorithm for the network embedding as matrix factorization.\n");
    printf("*************************************************************************************\n\n");
    printf("Input matrix file: %s\n\n", filename);
    printf("argv:%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n", argv[0], argv[2], argv[4], argv[6], argv[8], argv[10], argv[12], argv[14], argv[16], argv[18], argv[20], argv[22], argv[24], argv[26], argv[28], argv[30]);
    struct timeval start_timeval, end_timeval;
    gettimeofday(&start_timeval, NULL);
    FILE *fid = fopen(filename, "r");
    mat_coo *A = coo_matrix_new(n, n, nnz);
    A->nnz = nnz;
    double vol = 0;
    for (i = 0; i < nnz; i++)
    {
        int ii, jj;
        double kk;
        fscanf(fid, " (%d, %d) %lf\n", &ii, &jj, &kk);
        A->rows[i] = jj + 1;
        A->cols[i] = ii + 1;
        A->values[i] = kk;
        vol += kk;
    }
    fclose(fid);
    mat_csr *D = csr_matrix_new();
    csr_init_from_coo(D, A);
    coo_matrix_delete(A);
    netmf(D, vol / b);
    gettimeofday(&end_timeval, NULL);
    printf("the total time of single-pass sketch svd for netmf %f seconds\n", get_seconds_frac(start_timeval, end_timeval));
    csr_matrix_delete(D);
    return 0;
}
