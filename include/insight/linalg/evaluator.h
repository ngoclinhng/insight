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

// Evaluate a `normal` binary expression.
template<typename E1, typename E2, typename F>
struct evaluator<
  binary_expression<E1, E2, F>,
  typename
  std::enable_if<
    !is_special_binary_expression<binary_expression<E1, E2, F>>::value,
    void>::type> {
  using value_type = typename binary_expression<E1, E2, F>::value_type;

  inline static void assign(const binary_expression<E1, E2, F>& expr,
                            value_type* buffer) {
    std::copy(expr.begin(), expr.end(), buffer);
  }

  inline static void add(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  inline static void sub(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  inline static void mul(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  inline static void div(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};


// Evaluate a `normal` unary expression.
template<typename E, typename F>
struct evaluator<
  unary_expression<E, F>,
  typename
  std::enable_if<
    !is_special_unary_expression<unary_expression<E, F> >::value,
    void>::type> {
  using value_type = typename unary_expression<E, F>::value_type;

  inline static void assign(const unary_expression<E, F>& expr,
                            value_type* buffer) {
    std::copy(expr.begin(), expr.end(), buffer);
  }

  inline static void add(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  inline static void sub(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  inline static void mul(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  inline static void div(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// a * x: where `x` is a floating-point, dense vector and `a` is a
// floating-point scalar.
template<typename E1, typename E2, typename F>
struct evaluator<
  binary_expression<E1, E2, F>,
  typename std::enable_if<
    is_fd_times_scalar<binary_expression<E1, E2, F> >::value,
    void>::type> {
  using value_type = typename binary_expression<E1, E2, F>::value_type;

  inline static void assign(const binary_expression<E1, E2, F>& expr,
                            value_type* buffer) {
    // TODO(Linh): Benchmark carefully to make sure that two BLAS steps
    // actully beat the single, simple for-loop?
    // std::copy(expr.begin(), expr.end(), buffer);
    std::copy(expr.e.begin(), expr.e.end(), buffer);
    internal::insight_scal(expr.size(), expr.scalar, buffer);
  }

  // y += a * x
  inline static void add(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    internal::insight_axpy(expr.size(), expr.scalar, expr.e.begin(), buffer);
  }

  // y -= a * x
  inline static void sub(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    internal::insight_axpy(expr.size(), -expr.scalar, expr.e.begin(), buffer);
  }

  // y *= a * x
  inline static void mul(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= a * x
  inline static void div(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// x / a: where `x` is a floating-point. dense vector and `a` is a floating
// point scalar.
template<typename E1, typename E2, typename F>
struct evaluator<
  binary_expression<E1, E2, F>,
  typename std::enable_if<
    is_fd_div_scalar<binary_expression<E1, E2, F> >::value,
    void>::type> {
  using value_type = typename binary_expression<E1, E2, F>::value_type;

  // y = x/a.
  inline static void assign(const binary_expression<E1, E2, F>& expr,
                            value_type* buffer) {
    // TODO(Linh): Benchmark carefully to make sure that two BLAS steps
    // actully beat the single, simple for-loop?
    // std::copy(expr.begin(), expr.end(), buffer);
    std::copy(expr.e.begin(), expr.e.end(), buffer);
    internal::insight_scal(expr.size(), value_type(1.0)/expr.scalar, buffer);
  }

  // y += x/a
  inline static void add(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    internal::insight_axpy(expr.size(), value_type(1.0)/expr.scalar,
                           expr.e.begin(), buffer);
  }

  // y -= x/a
  inline static void sub(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    internal::insight_axpy(expr.size(), value_type(-1.0)/expr.scalar,
                           expr.e.begin(), buffer);
  }

  // y *= x/a
  inline static void mul(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= x/a
  inline static void div(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// x ? y: where `x`, `y` are floating-point, dense vector and `?` is one of
// addition, substraction, multiplication, or division operation.
template<typename E1, typename E2, typename F>
struct evaluator<
  binary_expression<E1, E2, F>,
  typename std::enable_if<
    is_fd_elemwise_op_fd<binary_expression<E1, E2, F> >::value,
    void>::type> {
  using value_type = typename binary_expression<E1, E2, F>::value_type;
  using functor_type = typename binary_expression<E1, E2, F>::functor_type;

  // z = x ? y.
  //
  // TODO(Linh): This is NOT scalable! Consider doing something like
  // this:
  //  binary_functor_traits<functor_type>::apply(expr, buffer);
  inline static void assign(const binary_expression<E1, E2, F>& expr,
                            value_type* buffer) {
    if (std::is_same<functor_type, std::plus<value_type> >::value) {
      internal::insight_add(expr.size(), expr.e1.begin(), expr.e2.begin(),
                            buffer);
    } else if (std::is_same<functor_type, std::minus<value_type> >::value) {
      internal::insight_sub(expr.size(), expr.e1.begin(), expr.e2.begin(),
                            buffer);
    } else if (std::is_same<functor_type,
               std::multiplies<value_type> >::value) {
      internal::insight_mul(expr.size(), expr.e1.begin(), expr.e2.begin(),
                            buffer);
    } else if (std::is_same<functor_type,
               std::divides<value_type> >::value) {  // NOLINT
      internal::insight_div(expr.size(), expr.e1.begin(), expr.e2.begin(),
                            buffer);
    } else {
      // Unknown functor type, fallback to default.
      std::copy(expr.begin(), expr.end(), buffer);
    }
  }

  // z += x ? y.
  inline static void add(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= x ? y.
  inline static void sub(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= x ? y.
  inline static void mul(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= x ? y.
  inline static void div(const binary_expression<E1, E2, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};


template<typename E, typename F>
struct evaluator<
  unary_expression<E, F>,
  typename std::enable_if<
    is_unary_functor_of_fd<unary_expression<E, F> >::value,
    void>::type> {
  using value_type = typename unary_expression<E, F>::value_type;
  using functor_type = typename unary_expression<E, F>::functor_type;

  // TODO(Linh): This is NOT scalable! Consider doing something like
  // this:
  //  unary_functor_traits<functor_type>::apply(expr, buffer);
  inline static void assign(const unary_expression<E, F>& expr,
                            value_type* buffer) {
    if (std::is_same<functor_type,
        unary_functor::sqrt<value_type> >::value) {
      internal::insight_sqrt(expr.e.size(), expr.e.begin(), buffer);
    } else if (std::is_same<functor_type,
               unary_functor::exp<value_type> >::value) {
      internal::insight_exp(expr.e.size(), expr.e.begin(), buffer);
    } else if (std::is_same<functor_type,
               unary_functor::log<value_type> >::value) {
      internal::insight_log(expr.e.size(), expr.e.begin(), buffer);
    } else {
      // Fallback to default.
      std::copy(expr.begin(), expr.end(), buffer);
    }
  }

  inline static void add(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  inline static void sub(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  inline static void mul(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  inline static void div(const unary_expression<E, F>& expr,
                         value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_EVALUATOR_H_
