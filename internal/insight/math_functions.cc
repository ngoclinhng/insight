// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/internal/port.h"

#ifdef INSIGHT_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
  #include <cblas.h>
}
#endif

#include "insight/internal/math_functions.h"

namespace insight {
namespace internal {

// Y <- alpha * X + Y

template<>
void axpy<float>(const int N, const float alpha, const float* X, float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template<>
void axpy<double>(const int N, const double alpha, const double* X,
                  double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

}  // namespace internal
}  // namespace insight
