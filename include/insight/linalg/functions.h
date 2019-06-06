// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_FUNCTIONS_H_
#define INCLUDE_INSIGHT_LINALG_FUNCTIONS_H_

#include <cmath>

#include "insight/linalg/unary_functor.h"
#include "glog/logging.h"

namespace insight {

// Forward declarations
// TODO(Linh): Should we forward declaration or using include instead?
template<typename E, typename F> struct unary_expression;
template<typename Derived> struct vector_expression;
template<typename Derived> struct matrix_expression;
template<typename E> struct transpose_expression;

// TODO(Linh): This is too odd. See `dot_expression.h` file for
// reference.
template<typename M, typename V, typename Enable>
struct dot_expression;


// Transcendental functions on vectors.

template<typename E>
inline
unary_expression<E, unary_functor::sqrt<typename E::value_type> >
sqrt(const vector_expression<E>& e) {
  return unary_expression<
    E,
    unary_functor::sqrt<typename E::value_type>
    >(e.self(), unary_functor::sqrt<typename E::value_type>());
}

template<typename E>
inline
unary_expression<E, unary_functor::exp<typename E::value_type> >
exp(const vector_expression<E>& e) {
  return unary_expression<
    E,
    unary_functor::exp<typename E::value_type>
    >(e.self(), unary_functor::exp<typename E::value_type>());
}

template<typename E>
inline
unary_expression<E, unary_functor::log<typename E::value_type> >
log(const vector_expression<E>& e) {
  return unary_expression<
    E,
    unary_functor::log<typename E::value_type>
    >(e.self(), unary_functor::log<typename E::value_type>());
}

// Transcendental functions on matrices.

template<typename E>
inline
unary_expression<E, unary_functor::sqrt<typename E::value_type> >
sqrt(const matrix_expression<E>& e) {
  return unary_expression<
    E,
    unary_functor::sqrt<typename E::value_type>
    >(e.self(), unary_functor::sqrt<typename E::value_type>());
}

template<typename E>
inline
unary_expression<E, unary_functor::exp<typename E::value_type> >
exp(const matrix_expression<E>& e) {
  return unary_expression<
    E,
    unary_functor::exp<typename E::value_type>
    >(e.self(), unary_functor::exp<typename E::value_type>());
}

template<typename E>
inline
unary_expression<E, unary_functor::log<typename E::value_type> >
log(const matrix_expression<E>& e) {
  return unary_expression<
    E,
    unary_functor::log<typename E::value_type>
    >(e.self(), unary_functor::log<typename E::value_type>());
}

// transpose.

// Transpose of a generic matrix expression.
template<typename E>
inline
transpose_expression<E> transpose(const matrix_expression<E>& expr) {
  return transpose_expression<E>(expr.self());
}

// Transpose of a generic vector expression.
template<typename E>
inline
transpose_expression<E> transpose(const vector_expression<E>& expr) {
  return transpose_expression<E>(expr.self());
}

// dot.

// generic matrix-vector multiplication.
template<typename M, typename V>
inline
dot_expression<M, V, void>
dot(const matrix_expression<M>& me, const vector_expression<V>& ve) {
  CHECK_EQ(me.self().num_cols(), ve.self().num_rows())
      << "mismatched dimensions for matrix-vector multiplication";
  return dot_expression<M, V, void>(me.self(), ve.self());
}

// generic matrix-matrix multiplication.
template<typename M1, typename M2>
inline
dot_expression<M1, M2, void>
dot(const matrix_expression<M1>& m1, const matrix_expression<M2>& m2) {
  CHECK_EQ(m1.self().num_cols(), m2.self().num_rows())
      << "mismatched dimensions for matrix-matrix multiplication";
  return dot_expression<M1, M2, void>(m1.self(), m2.self());
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_FUNCTIONS_H_
