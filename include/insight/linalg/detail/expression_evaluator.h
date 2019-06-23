// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_EVALUATOR_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_EVALUATOR_H_

#include <algorithm>

#include "insight/linalg/detail/expression_traits.h"
#include "insight/internal/math_functions.h"

namespace insight {
namespace linalg_detail {

template<typename E, typename Category = expression_category::normal>
struct expression_evaluator {
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

template<typename E>
struct expression_evaluator<E, expression_category::ax>{
  using value_type = typename E::value_type;

  // y = ax
  inline static void assign(const E& expr, value_type* buffer) {
    // TODO(Linh): Benchmark carefully to make sure that the two BLAS steps
    // actually beat the performance of a single copy call.
    std::copy(expr.e.begin(), expr.e.end(), buffer);
    internal::insight_scal(expr.size(), expr.scalar, buffer);
  }

  // y += ax
  inline static void add(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), expr.scalar, expr.e.begin(),
                           buffer);
  }

  // y -= ax.
  inline static void sub(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), -expr.scalar, expr.e.begin(),
                           buffer);
  }

  // y *= ax
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= ax
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// Evaluate a binary expression of the form `x + y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct expression_evaluator<E, expression_category::xpy> {
  using value_type = typename E::value_type;

  // z = x + y
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_add(expr.size(), expr.e1.begin(), expr.e2.begin(),
                          buffer);
  }

  // z += x + y
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= x + y
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= x + y
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= x + y
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// Evaluate a binary expression of the form `x - y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct expression_evaluator<E, expression_category::xmy> {
  using value_type = typename E::value_type;

  // z = x - y
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_sub(expr.size(), expr.e1.begin(), expr.e2.begin(),
                          buffer);
  }

  // z += x - y
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= x - y
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= x - y
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= x - y
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// Evaluate a binary expression of the form `x * y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct expression_evaluator<E, expression_category::xty> {
  using value_type = typename E::value_type;

  // z = x * y
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_mul(expr.size(), expr.e1.begin(), expr.e2.begin(),
                          buffer);
  }

  // z += x * y
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= x * y
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= x * y
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= x * y
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// Evaluate a binary expression of the form `x / y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct expression_evaluator<E, expression_category::xdy> {
  using value_type = typename E::value_type;

  // z = x/y
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_div(expr.size(), expr.e1.begin(), expr.e2.begin(),
                          buffer);
  }

  // z += x/y
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= x/y
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= x/y
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= x/y
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// Evaluate sqrt(x) where x is a floating-point, dense matrix/vector.
template<typename E>
struct expression_evaluator<E, expression_category::sqrt> {
  using value_type = typename E::value_type;

  // y = sqrt(x)
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_sqrt(expr.size(), expr.e.begin(), buffer);
  }

  // y += sqrt(x)
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // y -= sqrt(x)
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // y *= sqrt(x)
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= sqrt(x)
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// Evaluate exp(x) where x is a floating-point, dense matrix/vector.
template<typename E>
struct expression_evaluator<E, expression_category::exp> {
  using value_type = typename E::value_type;

  // y = exp(x)
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_exp(expr.size(), expr.e.begin(), buffer);
  }

  // y += exp(x)
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // y -= exp(x)
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // y *= exp(x)
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= exp(x)
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};


// Evaluate log(x) where x is a floating-point, dense matrix/vector.
template<typename E>
struct expression_evaluator<E, expression_category::log> {
  using value_type = typename E::value_type;

  // y = log(x)
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_log(expr.size(), expr.e.begin(), buffer);
  }

  // y += log(x)
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // y -= log(x)
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // y *= log(x)
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= log(x)
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_EVALUATOR_H_
