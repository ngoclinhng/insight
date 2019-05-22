// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_MATRIX_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_MATRIX_EXPRESSION_H_

#include <type_traits>

#include "insight/linalg/fmatrix_base.h"


namespace insight {

// template<typename L, typename R> struct mul_expr;

template<typename Matrix, typename T>
struct mul_expr<
  fmatrix_base<Matrix, T>,
  typename std::enable_if<std::is_floating_point<T>::value, T>::type> {
  const fmatrix_base<Matrix, T>& m;
  T scalar;

  mul_expr(const fmatrix_base<Matrix, T>& m, T scalar)
      : m(m), scalar(scalar) {}

  const Matrix& matrix_ref() const { return m.self(); }
};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_MATRIX_EXPRESSION_H_
