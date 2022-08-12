#include <math.h>
#include "matrix_vector_functions_intel_mkl_ext.h"

/*[L, ~] = lu(A) as in MATLAB*/
void LUfraction(mat *A, mat *L);

/*[U, S, V] = eigSVD(A)*/
void eigSVD(mat *A, mat *U, mat *S, mat *V);

void freigs(mat_csr *A, mat *U, mat *S, int k, int q, int s);

void freigs_convex(mat_csr *A, mat *U, mat *S, int k, int q, int s);

void eigs(mat_csr *A, mat *U, mat *S, int k, int q, int s);
