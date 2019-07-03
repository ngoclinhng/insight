// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_ASSIGN_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_ASSIGN_H_

#include <algorithm>

#include "insight/linalg/detail/special_expression_traits.h"
#include "insight/linalg/detail/blas_routines.h"

namespace insight {
namespace linalg_detail {

template<typename E> struct is_special_assignable
    : public std::conditional<
  special_expression::is_ax<E>::value ||
  special_expression::is_xpy<E>::value ||
  special_expression::is_xmy<E>::value ||
  special_expression::is_xty<E>::value ||
  special_expression::is_xdy<E>::value ||
  special_expression::is_sqrt_of_x<E>::value ||
  special_expression::is_exp_of_x<E>::value ||
  special_expression::is_log_of_x<E>::value ||
  special_expression::is_matmul_aAbx<E>::value ||
  special_expression::is_matmul_aAtbx<E>::value ||
  special_expression::is_alpha_times_matmul_aAbx<E>::value ||
  special_expression::is_alpha_times_matmul_aAtbx<E>::value,
  std::true_type,
  std::false_type>::type{};

namespace special_expression {

// buffer = ax
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_ax<E>::value>::type* = 0) {
  std::copy(expr.e.begin(), expr.e.end(), buffer);
  blas_scal(expr.size(), expr.scalar, buffer);
}

// buffer = x + y.
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_xpy<E>::value>::type* = 0) {
  blas_add(expr.size(), expr.e1.begin(), expr.e2.begin(),
           buffer);
}

// buffer = x - y.
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_xmy<E>::value>::type* = 0) {
  blas_sub(expr.size(), expr.e1.begin(), expr.e2.begin(),
           buffer);
}

// buffer = x * y
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_xty<E>::value>::type* = 0) {
  blas_mul(expr.size(), expr.e1.begin(), expr.e2.begin(),
           buffer);
}

// buffer = x / y
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_xdy<E>::value>::type* = 0) {
  blas_div(expr.size(), expr.e1.begin(), expr.e2.begin(),
           buffer);
}

// buffer = sqrt(x)
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_sqrt_of_x<E>::value>::type* = 0) {
  blas_sqrt(expr.size(), expr.e.begin(), buffer);
}

// buffer = exp(x)
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_exp_of_x<E>::value>::type* = 0) {
  blas_exp(expr.size(), expr.e.begin(), buffer);
}

// buffer = log(x)
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<is_log_of_x<E>::value>::type* = 0) {
  blas_log(expr.size(), expr.e.begin(), buffer);
}

// buffer = matmul(aA,bx)
template<typename M, typename V>
inline
void assign(const matmul_expression<M, V>& expr,
            typename matmul_expression<M, V>::value_type* buffer,
            typename std::enable_if<
            is_matmul_aAbx<matmul_expression<M, V>>::value>::type* = 0) {
  using value_type = typename matmul_expression<M, V>::value_type;
  matmul_aAbx_wrapper<M, V> wrapper(expr);
  blas_gemv(CblasNoTrans,
            wrapper.A_row_count(),
            wrapper.A_col_count(),
            wrapper.a() * wrapper.b(),
            wrapper.A(),
            wrapper.x(),
            value_type()/*zero*/,
            buffer);
}

// buffer = matmul(aA.t(),bx)
template<typename M, typename V>
inline
void assign(const matmul_expression<M, V>& expr,
            typename matmul_expression<M, V>::value_type* buffer,
            typename std::enable_if<
            is_matmul_aAtbx<matmul_expression<M, V>>::value>::type* = 0) {
  using value_type = typename matmul_expression<M, V>::value_type;
  matmul_aAtbx_wrapper<M, V> wrapper(expr);
  blas_gemv(CblasTrans,
            wrapper.A_row_count(),
            wrapper.A_col_count(),
            wrapper.a() * wrapper.b(),
            wrapper.A(),
            wrapper.x(),
            value_type()/*zero*/,
            buffer);
}

// buffer = alpha * matmul(aA, bx)
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<
            is_alpha_times_matmul_aAbx<E>::value>::type* = 0) {
  using value_type = typename E::value_type;
  auto wrapper = make_matmul_aAbx_wrapper(expr.e);
  blas_gemv(CblasNoTrans,
            wrapper.A_row_count(),
            wrapper.A_col_count(),
            wrapper.a() * wrapper.b() * expr.scalar,
            wrapper.A(),
            wrapper.x(),
            value_type()/*zero*/,
            buffer);
}

// buffer = alpha * matmul(aA.t(), bx)
template<typename E>
inline
void assign(const E& expr, typename E::value_type* buffer,
            typename std::enable_if<
            is_alpha_times_matmul_aAtbx<E>::value>::type* = 0) {
  using value_type = typename E::value_type;
  auto wrapper = make_matmul_aAtbx_wrapper(expr.e);
  blas_gemv(CblasTrans,
            wrapper.A_row_count(),
            wrapper.A_col_count(),
            wrapper.a() * wrapper.b() * expr.scalar,
            wrapper.A(),
            wrapper.x(),
            value_type()/*zero*/,
            buffer);
}

}  // namespace special_expression
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_ASSIGN_H_
