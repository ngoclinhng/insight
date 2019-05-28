// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_EVALUATE_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_EVALUATE_EXPRESSION_H_

#include "insight/linalg/type_traits.h"
#include "insight/internal/math_functions.h"

namespace insight {

// Evaluate a generic binary expression
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_normal_bin_expr<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  using size_type = typename binary_expr<E1, E2, Function>::size_type;

  matrix<value_type> temp(expr.shape());

  for (size_type i = 0; i < temp.size(); ++i) {
    temp[i] = expr[i];
  }
  return temp;
}

// aA.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_fd_times_scalar<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp = expr.e;
  internal::insight_scal(temp.size(), expr.scalar, temp.begin());
  return temp;
}

// A/a.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_fd_div_scalar<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp = expr.e;
  internal::insight_scal(temp.size(), value_type(1.0)/expr.scalar,
                         temp.begin());
  return temp;
}

// A + a.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_fd_plus_scalar<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp(expr.shape(), expr.scalar);
  internal::insight_axpy(temp.size(), value_type(1.0), expr.e.begin(),
                         temp.begin());
  return temp;
}

// A - a
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_fd_minus_scalar<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp(expr.shape(), -expr.scalar);
  internal::insight_axpy(temp.size(), value_type(1.0), expr.e.begin(),
                         temp.begin());
  return temp;
}

// a - A.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_scalar_minus_fd<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp(expr.shape(), expr.scalar);
  internal::insight_axpy(temp.size(), value_type(-1.0), expr.e.begin(),
                         temp.begin());
  return temp;
}

// aA + b.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_alpha_fd_plus_beta<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp(expr.shape(), expr.scalar);
  internal::insight_axpy(temp.size(), expr.e.scalar, expr.e.e.begin(),
                         temp.begin());
  return temp;
}

// aA - b.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_alpha_fd_minus_beta<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp(expr.shape(), -expr.scalar);
  internal::insight_axpy(temp.size(), expr.e.scalar, expr.e.e.begin(),
                         temp.begin());
  return temp;
}

// b - aA.
template<typename E1, typename E2, typename Function>
inline
typename
std::enable_if<
  is_beta_minus_alpha_fd<binary_expr<E1, E2, Function> >::value,
  matrix<typename binary_expr<E1, E2, Function>::value_type>
  >::type
evaluate_expression(const binary_expr<E1, E2, Function>& expr) {
  using value_type = typename binary_expr<E1, E2, Function>::value_type;
  matrix<value_type> temp(expr.shape(), expr.scalar);
  internal::insight_axpy(temp.size(), -expr.e.scalar, expr.e.e.begin(),
                         temp.begin());
  return temp;
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_EVALUATE_EXPRESSION_H_
