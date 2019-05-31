// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_EVALUATOR_H_
#define INCLUDE_INSIGHT_LINALG_EVALUATOR_H_

#include <algorithm>

#include "insight/linalg/type_traits.h"
#include "insight/internal/math_functions.h"

namespace insight {

template<typename E, typename Enable = void> struct evaluator;

// Evaluate a `normal` vector expression.
template<typename E>
struct evaluator<
  vector_expression<E>,
  typename std::enable_if<!is_special_vector_expression<E>::value,
                          void>::type> {
  using value_type = typename E::value_type;

  inline static void assign(const E& expr, value_type* buffer) {
    std::copy(expr.begin(), expr.end(), buffer);
  }

  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};


// a * x: where `x` is a floating-point, dense vector and `a` is a
// floating-point scalar.
template<typename E>
struct evaluator<vector_expression<E>,
                 typename std::enable_if<is_fd_times_scalar<E>::value,
                                         void>::type> {
  using value_type = typename E::value_type;

  inline static void assign(const E& expr, value_type* buffer) {
    // TODO(Linh): Benchmark carefully to make sure that two BLAS steps
    // actully beat the single, simple for-loop?
    // std::copy(expr.begin(), expr.end(), buffer);
    std::copy(expr.e.begin(), expr.e.end(), buffer);
    internal::insight_scal(expr.size(), expr.scalar, buffer);
  }

  // y += a * x
  inline static void add(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), expr.scalar, expr.e.begin(), buffer);
  }

  // y -= a * x
  inline static void sub(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), -expr.scalar, expr.e.begin(), buffer);
  }

  // y *= a * x
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= a * x
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// x / a: where `x` is a floating-point. dense vector and `a` is a floating
// point scalar.
template<typename E>
struct evaluator<vector_expression<E>,
                 typename std::enable_if<is_fd_div_scalar<E>::value,
                                         void>::type> {
  using value_type = typename E::value_type;

  // y = x/a.
  inline static void assign(const E& expr, value_type* buffer) {
    // TODO(Linh): Benchmark carefully to make sure that two BLAS steps
    // actully beat the single, simple for-loop?
    // std::copy(expr.begin(), expr.end(), buffer);
    std::copy(expr.e.begin(), expr.e.end(), buffer);
    internal::insight_scal(expr.size(), value_type(1.0)/expr.scalar, buffer);
  }

  // y += x/a
  inline static void add(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), value_type(1.0)/expr.scalar,
                           expr.e.begin(), buffer);
  }

  // y -= x/a
  inline static void sub(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), value_type(-1.0)/expr.scalar,
                           expr.e.begin(), buffer);
  }

  // y *= x/a
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= x/a
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};


}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_EVALUATOR_H_
