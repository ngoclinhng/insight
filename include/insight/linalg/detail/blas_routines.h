// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_BLAS_ROUTINES_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_BLAS_ROUTINES_H_

#include "insight/internal/port.h"

#ifdef INSIGHT_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif defined(INSIGHT_USE_MKL)
#include <mkl.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

namespace insight {
namespace linalg_detail {

// X <- α * X.
template<typename T>
void blas_scal(const int N, const T alpha, T* X);

// Y <- alpha * X + Y
template<typename T>
void blas_axpy(const int N, const T alpha, const T* X, T* Y);

// Y <- alpha * X + beta * Y
template<typename T>
void blas_axpby(const int N, const T alpha, const T* X, const T beta,
           T* Y);

// y <- αAx + βy.
template<typename T>
void blas_gemv(const CBLAS_TRANSPOSE TransA,
                  const int M,
                  const int N,
                  const T alpha,
                  const T* A,
                  const T* x,
                  const T beta,
                  T* y);

// This function multiplies A * B and multiplies the resulting matrix by
// alpha. It then multiplies matrix C by beta. It stores the sum of these
// two products in matrix C.
//
// Thus, it calculates either
//
//    C←αAB + βC
//
//    or
//
//    C←αBA + βC
//
//    with optional use of transposed forms of A, B, or both.
template<typename T>
void blas_gemm(const CBLAS_TRANSPOSE TransA,
                  const CBLAS_TRANSPOSE TransB,
                  const int M,
                  const int N,
                  const int K,
                  const T alpha,
                  const T* A,
                  const T* B,
                  const T beta,
                  T* C);

// Computes the L2 norm (Euclidian length) of a vector.
template<typename T>
T blas_nrm2(const int N, const T* X);

// Compute the dot product of two vectors.
template<typename T>
T blas_dot(const int N, const T* X, const T* Y);

// Adds two vectors: Z = X + Y element-wise.
template<typename T>
void blas_add(const int N, const T* X, const T* Y, T* Z);

// Performs element by element subtraction of vector Y from vector X, and
// stores the result in Z.
template<typename T>
void blas_sub(const int N, const T* X, const T* Y, T* Z);

// Performs element by element multiplication of vector X and vector Y, and
// stores the result in Z.
template<typename T>
void blas_mul(const int N, const T* X, const T* Y, T* Z);

// Performs element by element division of vector X by vector Y, and stores
// the result in Z.
template<typename T>
void blas_div(const int N, const T* X, const T* Y, T* Z);

// Performs element by element square root of the vector X, and stores the
// result in vector Y.
template<typename T>
void blas_sqrt(const int N, const T* X, T* Y);

// Calculates e raised to the power of each element in a vector.
template<typename T>
void blas_exp(const int N, const T* X, T* Y);

// Calculate s the natural logarithm for each element in the vector X, and
// stores the result in vector Y.
template<typename T>
void blas_log(const int N, const T* X, T* Y);
}  // namespace linalg_detail
}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_BLAS_ROUTINES_H_
