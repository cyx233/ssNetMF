#include "freigs.h"

extern void dsaupd_(int *ido, char *bmat, int *n, char *which,
                    int *nev, double *tol, double *resid, int *ncv,
                    double *v, int *ldv, int *iparam, int *ipntr,
                    double *workd, double *workl, int *lworkl,
                    int *info);

extern void dseupd_(int *rvec, char *All, int *select, double *d,
                    double *v, int *ldv, double *sigma,
                    char *bmat, int *n, char *which, int *nev,
                    double *tol, double *resid, int *ncv, double *v2,
                    int *ldv2, int *iparam, int *ipntr, double *workd,
                    double *workl, int *lworkl, int *ierr);

void eigs(mat_csr *A, mat *U, mat *S, int k, int q, int s)
{
    // dimension of the matrix
    int n = A->ncols;

    // number of eigenvalues to calculate
    int nev = k + s;

    // reverse communication parameter, must be zero on first iteration
    int ido = 0;
    // standard eigenvalue problem A*x=lambda*x
    char bmat = 'I';
    // calculate the smallest algebraic eigenvalue
    char which[3] = "LA";
    // calculate until machine precision
    double tol = 1E-6;

    // the residual vector
    double *resid = calloc(n, sizeof(double));

    // the number of columns in v: the number of lanczos vector
    // generated at each iteration, ncv <= n
    int ncv = 2 * nev;

    if (ncv < 20)
        ncv = 20;

    // v containts the lanczos basis vectors
    int ldv = n;
    double *v = calloc(ldv * ncv, sizeof(double));

    int *iparam = calloc(11, sizeof(int));
    iparam[0] = 1; // Specifies the shift strategy (1->exact)
    iparam[2] = q; // Maximum number of iterations
    iparam[6] = 1; /* Sets the mode of dsaupd.
                        1 is exact shifting,
                        2 is user-supplied shifts,
                        3 is shift-invert mode,
                        4 is buckling mode,
                        5 is Cayley mode. */

    int *ipntr = calloc(11, sizeof(int)); /* Indicates the locations in the work array workd
                                            where the input and output vectors in the
                                            callback routine are located. */

    // array used for reverse communication
    double *workd = calloc(3 * n, sizeof(double));

    int lworkl = ncv * (ncv + 8); /* Length of the workl array */
    double *workl = calloc(lworkl, sizeof(double));

    // info = 0: random start vector is used
    int info = 0; /* Passes convergence information out of the iteration
                     routine. */

    // rvec == 0 : calculate only eigenvalue
    // rvec > 0 : calculate eigenvalue and eigenvector
    int rvec = 1;

    // when All='All', this is used as workspace to reorder the eigenvectors
    int *select = calloc(ncv, sizeof(int));

    // This vector will return the eigenvalues from the second routine, dseupd.
    double *d = calloc(nev, sizeof(double));

    double *z = calloc(n * nev, sizeof(double));

    // not used if iparam[6] == 1
    double sigma;

    mat *M1 = malloc(sizeof(mat));
    M1->nrows = n;
    M1->ncols = 1;
    mat *M2 = malloc(sizeof(mat));
    M2->nrows = n;
    M2->ncols = 1;
    printf("Begin iter\n");
    int cnt = 0;
    while (ido != 99)
    {
        dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
        if (ido == 1 || ido == -1)
        {
            // matrix-vector multiplication
            M1->d = workd + ipntr[0] - 1;
            M2->d = workd + ipntr[1] - 1;
            csr_matrix_matrix_mult(A, M1, M2);
        }
    }
    printf("End iter\n");
    free(M1);
    free(M2);

    if (info < 0)
        printf("Error with dsaupd, info = %d\n", info);
    else if (info == 1)
        printf("Maximum number of Lanczos iterations reached.\n");
    else if (info == 3)
        printf("No shifts could be applied during implicit Arnoldi update, try increasing NCV.\n");

    dseupd_(&rvec, "All", select, d, z, &ldv, &sigma, &bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

    printf("extract finish\n");

    if (info != 0)
        printf("Error with dseupd, info = %d\n", info);
    else
    {
        for (int i = nev - 1; i >= nev - k; --i)
            matrix_set_element(S, nev - 1 - i, 0, d[i]);
        for (int i = 0; i < n; ++i)
            for (int j = nev - 1; j >= nev - k; --j)
                matrix_set_element(U, i, nev - 1 - j, z[i + j * n]);
    }

    free(resid);
    free(v);
    free(iparam);
    free(ipntr);
    free(workd);
    free(workl);
    free(d);

    free(z);
    free(select);
}

/*[L, ~] = lu(A) as in MATLAB*/
void LUfraction(mat *A, mat *L)
{
    matrix_copy(L, A);
    int *ipiv = (int *)malloc(sizeof(int) * L->nrows);
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, L->nrows, L->ncols, L->d, L->nrows, ipiv);
    int i, j;
#pragma omp parallel private(i, j)
    {
#pragma omp for
        for (i = 0; i < L->ncols; i++)
        {
            for (j = 0; j < i; j++)
            {
                L->d[i * L->nrows + j] = 0;
            }
            L->d[i * L->nrows + i] = 1;
        }
    }

    {
        for (i = L->ncols - 1; i >= 0; i--)
        {
            int ipi = ipiv[i] - 1;
            for (j = 0; j < L->ncols; j++)
            {
                double temp = L->d[j * L->nrows + ipi];
                L->d[j * L->nrows + ipi] = L->d[j * L->nrows + i];
                L->d[j * L->nrows + i] = temp;
            }
        }
    }
}

/*[U, S, V] = eigSVD(A)*/
void eigSVD(mat *A, mat *U, mat *S, mat *V)
{
    matrix_transpose_matrix_mult(A, A, V);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', V->ncols, V->d, V->ncols, S->d);
    mat *V1 = matrix_new(V->ncols, V->ncols);
    matrix_copy(V1, V);
    int i, j;
#pragma omp parallel shared(V1, S) private(i, j)
    {
#pragma omp for
        for (i = 0; i < V1->ncols; i++)
        {
            S->d[i] = sqrt(S->d[i]);
            for (j = 0; j < V1->nrows; j++)
                V1->d[i * V1->nrows + j] /= S->d[i];
        }
    }
    mat *Uc = matrix_new(U->nrows, U->ncols);
    matrix_matrix_mult(A, V1, Uc);
    matrix_copy(U, Uc);
    matrix_delete(Uc);
    matrix_delete(V1);
}

typedef struct
{
    double eval;
    int e_ind;
    double realeval;
} nd;

int cmpfreigs(nd *a, nd *b)
{
    return a->eval < b->eval ? 1 : -1;
}

void freigs(mat_csr *A, mat *U, mat *S, int k, int q, int s)
{
    printf("Begin freigs\n");
    int l = k + s;
    printf("randomized eig have l:%d--q:%d\n", l, q);
    mat *Omega = matrix_new(A->ncols, l);
    mat *Y = matrix_new(A->nrows, l);
    mat *Q = matrix_new(A->nrows, l);
    mat *Q_tmp = matrix_new(A->nrows, l);
    mat *UU = matrix_new(l, l);
    mat *SS = matrix_new(l, 1);
    mat *VV = matrix_new(l, l);

    initialize_random_matrix(Omega);
    //initialize_random_large_matrix(Omega);

    mat *Q_tmp2 = matrix_new(A->nrows, l);
    csr_matrix_matrix_mult(A, Omega, Y);
    matrix_delete(Omega);
    eigSVD(Y, Q, SS, VV);
    int i;
    for (i = 0; i < q; i++)
    {
        csr_matrix_matrix_mult(A, Q, Q_tmp);
        csr_matrix_matrix_mult(A, Q_tmp, Q_tmp2);
        if (i < q - 1)
        {
            LUfraction(Q_tmp2, Q);
        }
        else
        {
            eigSVD(Q_tmp2, Q, SS, VV);
        }
    }
    matrix_delete(Q_tmp);
    matrix_delete(Q_tmp2);
    mat *B = matrix_new(A->nrows, l);
    mat *M = matrix_new(l, l);
    csr_matrix_matrix_mult(A, Q, B);
    matrix_transpose_matrix_mult(Q, B, M);
    //matrix_print(M);
    int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', M->ncols, M->d, M->ncols, SS->d);
    if (info != 0)
    {
        printf("Some Error happen in the eig,info:%d\n", info);
        exit(0);
    }
    int inds[k];
    for (i = s; i < k + s; i++)
    {
        inds[i - s] = k + s - (i - s) - 1;
        //inds[i-s] = i;
    }
    mat *UU2 = matrix_new(l, k);
    matrix_get_selected_columns(M, inds, UU2);
    matrix_get_selected_rows(SS, inds, S);
    matrix_matrix_mult(Q, UU2, U);

    matrix_delete(Q);
    matrix_delete(Y);
    matrix_delete(UU);
    matrix_delete(SS);
    matrix_delete(VV);
    matrix_delete(B);
    matrix_delete(M);
    matrix_delete(UU2);
}

void freigs_convex(mat_csr *A, mat *U, mat *S, int k, int q, int s)
{
    printf("Begin freigs_convex\n");
    int l = k + s;
    printf("randomized eig have l:%d--q:%d\n", l, q);
    mat *Omega = matrix_new(A->ncols, l);
    mat *Q = matrix_new(A->nrows, l);
    mat *Q_tmp = matrix_new(A->nrows, l);
    mat *Y = matrix_new(A->nrows, l);
    mat *UU = matrix_new(l, l);
    mat *SS = matrix_new(l, 1);
    mat *VV = matrix_new(l, l);
    initialize_random_matrix(Omega);
    //initialize_random_large_matrix(Omega);
    mat *Q_tmp2 = matrix_new(A->nrows, l);
    csr_matrix_matrix_mult(A, Omega, Y);
    matrix_delete(Omega);
    eigSVD(Y, Q, SS, VV);
    int i;
    for (i = 0; i < q; i++)
    {
        csr_matrix_matrix_mult(A, Q, Q_tmp);
        csr_matrix_matrix_mult(A, Q_tmp, Q_tmp2);
        if (i < q - 1)
        {
            LUfraction(Q_tmp2, Q);
        }
        else
        {
            eigSVD(Q_tmp2, Q, SS, VV);
        }
    }
    mat *B = matrix_new(A->nrows, l);
    csr_matrix_transpose_matrix_mult(A, Q, B);
    mat *C = matrix_new(A->nrows, 2 * l);
    int inds[l];
    for (int j = 0; j < l; j++)
        inds[j] = j;
    matrix_set_selected_columns(C, inds, Q);
    for (int j = 0; j < l; j++)
        inds[j] = j + l;
    matrix_set_selected_columns(C, inds, B);
    matrix_delete(B);
    mat *Ut = matrix_new(A->nrows, 2 * l);
    mat *T = matrix_new(2 * l, 2 * l);
    compact_QR_factorization(C, Ut, T);
    matrix_delete(C);
    mat *T1 = matrix_new(2 * l, l);
    mat *T2 = matrix_new(2 * l, l);
    for (int j = 0; j < l; j++)
        inds[j] = j;
    matrix_get_selected_columns(T, inds, T1);
    for (int j = 0; j < l; j++)
        inds[j] = j + l;
    matrix_get_selected_columns(T, inds, T2);
    matrix_delete(T);
    mat *St1 = matrix_new(2 * l, 2 * l);
    mat *St2 = matrix_new(2 * l, 2 * l);
    matrix_matrix_transpose_mult(T1, T2, St1);
    matrix_matrix_transpose_mult(T2, T1, St2);
    matrix_matrix_add(St1, St2);
    matrix_scale(St2, 0.5);
    matrix_delete(St1);
    mat *SS2 = matrix_new(2 * l, 1);
    int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', St2->ncols, St2->d, St2->ncols, SS2->d);
    if (info != 0)
    {
        printf("Some Error happen in the eig---info:%d\n", info);
        exit(0);
    }
    /*
    nd *S_eval = (nd*) calloc(2*l,sizeof(nd));
    for (i=0;i<2*l;i++){
        //matrix_set_element(SS2,i,0,fabs(matrix_get_element(SS2,i,0)));
        S_eval[i].realeval  = matrix_get_element(SS2,i,0);
        S_eval[i].eval=fabs(matrix_get_element(SS2,i,0));
        S_eval[i].e_ind=i;
    }
    qsort(S_eval,2*l,sizeof(S_eval[0]),cmpfreigs);
    mat *UU2 = matrix_new(2*l, k);
    vec *v_col = vector_new(2*l);
    for (i=0;i<k;i++){
        matrix_set_element(S,i,0,S_eval[i].realeval);
        v_col = vector_new(2*l);
        matrix_get_col(St2,S_eval[i].e_ind,v_col);
        matrix_set_col(UU2,i,v_col);
    }
    vector_delete(v_col);
    */

    int ind[k];
    for (i = k + 2 * s; i < 2 * (k + s); i++)
    {
        //ind[i-(k+2*s)] = i;
        ind[i - (k + 2 * s)] = 2 * (k + s) - (i - k - 2 * s) - 1;
    }
    mat *UU2 = matrix_new(2 * l, k);
    matrix_get_selected_columns(St2, ind, UU2);
    matrix_get_selected_rows(SS2, ind, S);
    matrix_matrix_mult(Ut, UU2, U);
    matrix_delete(Ut);
    matrix_delete(St2);
    matrix_delete(Q);
    matrix_delete(Q_tmp);
    matrix_delete(Q_tmp2);
    matrix_delete(Y);
    matrix_delete(UU);
    matrix_delete(SS);
    matrix_delete(VV);
    matrix_delete(UU2);
    matrix_delete(SS2);
}
