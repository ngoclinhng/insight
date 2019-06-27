// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_SUB_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_SUB_H_

#include "insight/linalg/detail/special_expression_traits.h"
#include "insight/internal/math_functions.h"

namespace insight {
namespace linalg_detail {

template<typename E> struct is_special_subtractable
    : public std::conditional<
  special_expression::is_ax<E>::value ||
  special_expression::is_matmul_aAbx<E>::value ||
  special_expression::is_matmul_aAtbx<E>::value,
  std::true_type,
  std::false_type>::type{};

namespace special_expression {

// buffer -= ax.
template<typename E>
inline
void sub(const E& expr, typename E::value_type* buffer,
         typename std::enable_if<is_ax<E>::value>::type* = 0) {
  internal::insight_axpy(expr.size(), -expr.scalar, expr.e.begin(),
                         buffer);
}

// buffer -= matmul(aA, bx)
template<typename M, typename V>
inline
void sub(const matmul_expression<M, V>& expr,
         typename matmul_expression<M, V>::value_type* buffer,
         typename std::enable_if<
         is_matmul_aAbx<matmul_expression<M, V> >::value>::type* = 0) {
  using value_type = typename matmul_expression<M, V>::value_type;
  matmul_aAbx_wrapper<M, V> wrapper(expr);
  internal::insight_gemv(CblasNoTrans,
                         wrapper.A_row_count(),
                         wrapper.A_col_count(),
                         -(wrapper.a() * wrapper.b()),
                         wrapper.A(),
                         wrapper.x(),
                         value_type(1.0),
                         buffer);
}

// buffer -= matmul(aA.t(), bx)
template<typename M, typename V>
inline
void sub(const matmul_expression<M, V>& expr,
         typename matmul_expression<M, V>::value_type* buffer,
         typename std::enable_if<
         is_matmul_aAtbx<matmul_expression<M, V> >::value>::type* = 0) {
  using value_type = typename matmul_expression<M, V>::value_type;
  matmul_aAtbx_wrapper<M, V> wrapper(expr);
  internal::insight_gemv(CblasTrans,
                         wrapper.A_row_count(),
                         wrapper.A_col_count(),
                         -(wrapper.a() * wrapper.b()),
                         wrapper.A(),
                         wrapper.x(),
                         value_type(1.0),
                         buffer);
}
}  // namespace special_expression
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_SUB_H_
