// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <cmath>

#include "insight/internal/math_functions.h"

namespace insight {
namespace internal {

// X <- α * X.

template<>
void insight_scal<float>(const int N, const float alpha, float* X) {
  cblas_sscal(N, alpha, X, 1);
}

template<>
void insight_scal<double>(const int N, const double alpha, double* X) {
  cblas_dscal(N, alpha, X, 1);
}

// Y <- alpha * X + Y

template<>
void insight_axpy<float>(const int N, const float alpha, const float* X,
                         float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template<>
void insight_axpy<double>(const int N, const double alpha, const double* X,
                          double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

// Y <- alpha * X + beta * Y

template<>
void insight_axpby<float>(const int N, const float alpha, const float* X,
                          const float beta, float* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  catlas_saxpby(N, alpha, X, 1, beta, Y, 1);
#else
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
#endif
}

template<>
void insight_axpby<double>(const int N, const double alpha, const double* X,
                           const double beta, double* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  catlas_daxpby(N, alpha, X, 1, beta, Y, 1);
#else
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
#endif
}

// y <- αAx + βy.

template<>
void insight_gemv<float>(const CBLAS_TRANSPOSE TransA,
                         const int M,
                         const int N,
                         const float alpha,
                         const float* A,
                         const float* x,
                         const float beta,
                         float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void insight_gemv<double>(const CBLAS_TRANSPOSE TransA,
                          const int M,
                          const int N,
                          const double alpha,
                          const double* A,
                          const double* x,
                          const double beta,
                          double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

// Gemm.

template<>
void insight_gemm<float>(const CBLAS_TRANSPOSE TransA,
                         const CBLAS_TRANSPOSE TransB,
                         const int M,
                         const int N,
                         const int K,
                         const float alpha,
                         const float* A,
                         const float* B,
                         const float beta,
                         float* C) {
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template<>
void insight_gemm<double>(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB,
                          const int M,
                          const int N,
                          const int K,
                          const double alpha,
                          const double* A,
                          const double* B,
                          const double beta,
                          double* C) {
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

// Computes the L2 norm (Euclidian length) of a vector.

template<>
float insight_nrm2<float>(const int N, const float* X) {
  return cblas_snrm2(N, X, 1);
}

template<>
double insight_nrm2<double>(const int N, const double* X) {
  return cblas_dnrm2(N, X, 1);
}

// Compute the dot product of two vectors.

template<>
float insight_dot<float>(const int N, const float* X, const float* Y) {
  return cblas_sdot(N, X, 1, Y, 1);
}

template<>
double insight_dot<double>(const int N, const double* X, const double* Y) {
  return cblas_ddot(N, X, 1, Y, 1);
}

// Adds two vectors: Z = X + Y element-wise.

template<>
void insight_add<float>(const int N, const float* X, const float* Y,
                        float* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vadd(X, 1, Y, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vsAdd(N, X, Y, Z);
#else
  // TODO(Linh): Should we use a simple for-loop or a combination of memcpy
  // and axpy?
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] + Y[i];
  }
#endif
}

template<>
void insight_add<double>(const int N, const double* X, const double* Y,
                         double* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vaddD(X, 1, Y, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vdAdd(N, X, Y, Z);
#else
  // TODO(Linh): Should we use a simple for-loop or a combination of memcpy
  // and axpy?
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] + Y[i];
  }
#endif
}

// Performs element by element subtraction of vector Y from vector X, and
// stores the result in Z.

template<>
void insight_sub<float>(const int N, const float* X, const float* Y,
                        float* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vsub(Y, 1, X, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vsSub(N, X, Y, Z);
#else
  // TODO(Linh): Should we use a simple for-loop or a combination of memcpy
  // and axpy?
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] - Y[i];
  }
#endif
}

template<>
void insight_sub<double>(const int N, const double* X, const double* Y,
                         double* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vsubD(Y, 1, X, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vdSub(N, X, Y, Z);
#else
  // TODO(Linh): Should we use a simple for-loop or a combination of memcpy
  // and axpy?
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] - Y[i];
  }
#endif
}

// Performs element by element multiplication of vector X and vector Y, and
// stores the result in Z.

template<>
void insight_mul<float>(const int N, const float* X, const float* Y,
                        float* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vmul(Y, 1, X, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vsMul(N, X, Y, Z);
#else
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] * Y[i];
  }
#endif
}

template<>
void insight_mul<double>(const int N, const double* X, const double* Y,
                         double* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vmulD(Y, 1, X, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vdMul(N, X, Y, Z);
#else
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] * Y[i];
  }
#endif
}

// Performs element by element division of vector a by vector b, and stores
// the result in Z.

template<>
void insight_div<float>(const int N, const float* X, const float* Y,
                        float* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vdiv(Y, 1, X, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vsDiv(N, X, Y, Z);
#else
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] / Y[i];
  }
#endif
}

template<>
void insight_div<double>(const int N, const double* X, const double* Y,
                         double* Z) {
#ifdef INSIGHT_USE_ACCELERATE
  vDSP_vdivD(Y, 1, X, 1, Z, 1, N);
#elif defined(INSIGHT_USE_MKL)
  vdDiv(N, X, Y, Z);
#else
  for (int i = 0; i < N; ++i) {
    Z[i] = X[i] / Y[i];
  }
#endif
}

// Performs element by element square root of the vector X, and stores the
// result in vector Y.

template<>
void insight_sqrt<float>(const int N, const float* X, float* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  int n = N;
  vvsqrtf(Y, X, &n);
#elif defined(INSIGHT_USE_MKL)
  vsSqrt(N, X, Y);
#else
  for (int i = 0; i < N; ++i) {
    Y[i] = std::sqrt(X[i]);
  }
#endif
}

template<>
void insight_sqrt<double>(const int N, const double* X, double* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  int n = N;
  vvsqrt(Y, X, &n);
#elif defined(INSIGHT_USE_MKL)
  vdSqrt(N, X, Y);
#else
  for (int i = 0; i < N; ++i) {
    Y[i] = std::sqrt(X[i]);
  }
#endif
}

// Calculates e raised to the power of each element in the vector X, and
// stores the result in vector Y.

template<>
void insight_exp<float>(const int N, const float* X, float* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  int n = N;
  vvexpf(Y, X, &n);
#elif defined(INSIGHT_USE_MKL)
  vsExp(N, X, Y);
#else
  for (int i = 0; i < N; ++i) {
    Y[i] = std::exp(X[i]);
  }
#endif
}

template<>
void insight_exp<double>(const int N, const double* X, double* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  int n = N;
  vvexp(Y, X, &n);
#elif defined(INSIGHT_USE_MKL)
  vdExp(N, X, Y);
#else
  for (int i = 0; i < N; ++i) {
    Y[i] = std::exp(X[i]);
  }
#endif
}

// Calculate s the natural logarithm for each element in the vector X, and
// stores the result in vector Y.

template<>
void insight_log<float>(const int N, const float* X, float* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  int n = N;
  vvlogf(Y, X, &n);
#elif defined(INSIGHT_USE_MKL)
  vsLn(N, X, Y);
#else
  for (int i = 0; i < N; ++i) {
    Y[i] = std::log(X[i]);
  }
#endif
}

template<>
void insight_log<double>(const int N, const double* X, double* Y) {
#ifdef INSIGHT_USE_ACCELERATE
  int n = N;
  vvlog(Y, X, &n);
#elif defined(INSIGHT_USE_MKL)
  vdLn(N, X, Y);
#else
  for (int i = 0; i < N; ++i) {
    Y[i] = std::log(X[i]);
  }
#endif
}

}  // namespace internal
}  // namespace insight
