// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_FUNCTIONS_H_
#define INCLUDE_INSIGHT_LINALG_FUNCTIONS_H_

#include <cmath>

#include "insight/linalg/detail/functors.h"
#include "glog/logging.h"

namespace insight {

namespace linalg_detail {

// Forward declarations
template<typename E, typename F> struct unary_expression;
template<typename Derived> struct vector_expression;
template<typename Derived> struct matrix_expression;
}  // linalg_detail

// Transcendental functions on vectors.

template<typename E>
inline
linalg_detail::unary_expression<E, linalg_detail::sqrt<typename E::value_type> >
sqrt(const linalg_detail::vector_expression<E>& e) {
  return linalg_detail::unary_expression<
    E,
    linalg_detail::sqrt<typename E::value_type>
    >(e.self(), linalg_detail::sqrt<typename E::value_type>());
}

template<typename E>
inline
linalg_detail::unary_expression<E, linalg_detail::exp<typename E::value_type> >
exp(const linalg_detail::vector_expression<E>& e) {
  return linalg_detail::unary_expression<
    E,
    linalg_detail::exp<typename E::value_type>
    >(e.self(), linalg_detail::exp<typename E::value_type>());
}

template<typename E>
inline
linalg_detail::unary_expression<E, linalg_detail::log<typename E::value_type> >
log(const linalg_detail::vector_expression<E>& e) {
  return linalg_detail::unary_expression<
    E,
    linalg_detail::log<typename E::value_type>
    >(e.self(), linalg_detail::log<typename E::value_type>());
}

// Transcendental functions on matrices.

template<typename E>
inline
linalg_detail::unary_expression<E, linalg_detail::sqrt<typename E::value_type> >
sqrt(const linalg_detail::matrix_expression<E>& e) {
  return linalg_detail::unary_expression<
    E,
    linalg_detail::sqrt<typename E::value_type>
    >(e.self(), linalg_detail::sqrt<typename E::value_type>());
}

template<typename E>
inline
linalg_detail::unary_expression<E, linalg_detail::exp<typename E::value_type> >
exp(const linalg_detail::matrix_expression<E>& e) {
  return linalg_detail::unary_expression<
    E,
    linalg_detail::exp<typename E::value_type>
    >(e.self(), linalg_detail::exp<typename E::value_type>());
}

template<typename E>
inline
linalg_detail::unary_expression<E, linalg_detail::log<typename E::value_type> >
log(const linalg_detail::matrix_expression<E>& e) {
  return linalg_detail::unary_expression<
    E,
    linalg_detail::log<typename E::value_type>
    >(e.self(), linalg_detail::log<typename E::value_type>());
}

// matmul.

// generic matrix-vector multiplication.
// template<typename M, typename V>
// inline
// matmul_expression<M, V, void>
// matmul(const linalg_detail::matrix_expression<M>& me, const linalg_detail::vector_expression<V>& ve) {
//   CHECK_EQ(me.self().num_cols(), ve.self().num_rows())
//       << "mismatched dimensions for matrix-vector multiplication";
//   return matmul_expression<M, V, void>(me.self(), ve.self());
// }

// // generic matrix-matrix multiplication.
// template<typename M1, typename M2>
// inline
// matmul_expression<M1, M2, void>
// matmul(const linalg_detail::matrix_expression<M1>& m1, const linalg_detail::matrix_expression<M2>& m2) {
//   CHECK_EQ(m1.self().num_cols(), m2.self().num_rows())
//       << "mismatched dimensions for matrix-matrix multiplication";
//   return matmul_expression<M1, M2, void>(m1.self(), m2.self());
// }

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_FUNCTIONS_H_
