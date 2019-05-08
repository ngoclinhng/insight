// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_MATH_FUNCTIONS_H_
#define INCLUDE_INSIGHT_INTERNAL_MATH_FUNCTIONS_H_

namespace insight {
namespace internal {

// Y <- alpha * X + Y
template<typename T>
void axpy(const int N, const T alpha, const T* X, T* Y);

// Y <- alpha * X + beta * Y
template<typename T>
void axpby(const int N, const T alpha, const T* X, const T beta,
           T* Y);

}  // namespace internal
}  // namespace insight

#endif  // INCLUDE_INSIGHT_INTERNAL_MATH_FUNCTIONS_H_
