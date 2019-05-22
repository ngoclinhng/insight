// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_OPERATOR_TIMES_H_
#define INCLUDE_INSIGHT_LINALG_OPERATOR_TIMES_H_

#include "insight/linalg/matrix_expression.h"

namespace insight {

template<typename Matrix, typename T>
inline
typename
std::enable_if<std::is_floating_point<T>::value,
               mul_expr<fmatrix_base<Matrix, T>, T> >::type
operator*(const fmatrix_base<Matrix, T>& m, T scalar) {
  return mul_expr<fmatrix_base<Matrix, T>, T>(m, scalar);
}

template<typename Matrix, typename T>
inline
typename
std::enable_if<std::is_floating_point<T>::value,
               mul_expr<fmatrix_base<Matrix, T>, T> >::type
operator*(T scalar, const fmatrix_base<Matrix, T>& m) {
  return mul_expr<fmatrix_base<Matrix, T>, T>(m, scalar);
}

template<typename Matrix, typename T>
inline
typename
std::enable_if<std::is_floating_point<T>::value,
               mul_expr<fmatrix_base<Matrix, T>, T> >::type
operator*(const mul_expr<fmatrix_base<Matrix, T>, T>& expr, T scalar) {
  return mul_expr<fmatrix_base<Matrix, T>, T>(expr.m, expr.scalar * scalar);
}

template<typename Matrix, typename T>
inline
typename
std::enable_if<std::is_floating_point<T>::value,
               mul_expr<fmatrix_base<Matrix, T>, T> >::type
operator*(T scalar, const mul_expr<fmatrix_base<Matrix, T>, T>& expr) {
  return mul_expr<fmatrix_base<Matrix, T>, T>(expr.m, expr.scalar * scalar);
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_OPERATOR_TIMES_H_
