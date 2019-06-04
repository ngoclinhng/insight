// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_EVALUATOR_H_
#define INCLUDE_INSIGHT_LINALG_EVALUATOR_H_

#include <algorithm>

#include "insight/linalg/type_traits/is_special_expression.h"
#include "insight/internal/math_functions.h"

namespace insight {

template<typename E, typename Enable = void> struct evaluator;

// Evaluate a normal, non-special expression.
template<typename E>
struct evaluator<E, typename std::enable_if< !is_special_expression<E>::value,void>::type > {  // NOLINT
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

// 1. ax
//
// Evaluate a binary expression of the form `a * x` where `a` is
// a floating-point scalar, and `x` is either a floating-point, dense vector
// or a floating-point, dense matrix.
template<typename E>
struct evaluator<E, typename std::enable_if<is_ax<E>::value, void>::type> {
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

// 2. xda
//
// Evaluate a binary expression of the form `x/a` where `a` is
// a floating-point scalar, and `x` is either a floating-point, dense vector
// or a floating-point, dense matrix.
template<typename E>
struct evaluator<E, typename std::enable_if<is_xda<E>::value, void>::type> {
  using value_type = typename E::value_type;

  // y = x/a
  inline static void assign(const E& expr, value_type* buffer) {
    // TODO(Linh): Benchmark carefully to make sure that the two BLAS steps
    // actually beat the performance of a single copy call.
    std::copy(expr.e.begin(), expr.e.end(), buffer);
    internal::insight_scal(expr.size(), value_type(1.0)/expr.scalar, buffer);
  }

  // y += x/a.
  inline static void add(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), value_type(1.0)/expr.scalar,
                           expr.e.begin(), buffer);
  }

  // y -= x/a.
  inline static void sub(const E& expr, value_type* buffer) {
    internal::insight_axpy(expr.size(), value_type(-1.0)/expr.scalar,
                           expr.e.begin(), buffer);
  }

  // y *= x/a.
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= x/a.
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};


// 3. xpy.
//
// Evaluate a binary expression of the form `x + y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct evaluator<E, typename std::enable_if<is_xpy<E>::value, void>::type> {
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

// 4. xmy.
//
// Evaluate a binary expression of the form `x - y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct evaluator<E, typename std::enable_if<is_xmy<E>::value, void>::type> {
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

// 5. xty.
//
// Evaluate a binary expression of the form `x * y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct evaluator<E, typename std::enable_if<is_xty<E>::value, void>::type> {
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

// 6. xdy.
//
// Evaluate a binary expression of the form `x / y` where `x` and `y` are
// either floating-point, dense vectors or floating-point, dense matrices.
template<typename E>
struct evaluator<E, typename std::enable_if<is_xdy<E>::value, void>::type> {
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

// 7. sqrt(x).
//
// Evaluate a unary expression of the form `sqrt(x)` where `x` is
// either a floating-point, dense vector or a floating-point, dense matrice.
template<typename E>
struct evaluator<E, typename std::enable_if<is_sqrt_x<E>::value,
                                            void>::type> {
  using value_type = typename E::value_type;

  // y = sqrt(x).
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_sqrt(expr.e.size(), expr.e.begin(), buffer);
  }

  // y += sqrt(x).
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= sqrt(x)
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= sqrt(x)
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= sqrt(x)
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// 8. exp(x).
//
// Evaluate a unary expression of the form `exp(x)` where `x` is
// either a floating-point, dense vector or a floating-point, dense matrice.
template<typename E>
struct evaluator<E, typename std::enable_if<is_exp_x<E>::value,
                                            void>::type> {
  using value_type = typename E::value_type;

  // y = exp(x).
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_exp(expr.e.size(), expr.e.begin(), buffer);
  }

  // y += exp(x).
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= exp(x)
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= exp(x)
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= exp(x)
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// 9. log(x).
//
// Evaluate a unary expression of the form `log(x)` where `x` is
// either a floating-point, dense vector or a floating-point, dense matrice.
template<typename E>
struct evaluator<E, typename std::enable_if<is_log_x<E>::value,
                                            void>::type> {
  using value_type = typename E::value_type;

  // y = log(x).
  inline static void assign(const E& expr, value_type* buffer) {
    internal::insight_log(expr.e.size(), expr.e.begin(), buffer);
  }

  // y += log(x).
  inline static void add(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ += e; });
  }

  // z -= log(x)
  inline static void sub(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ -= e; });
  }

  // z *= log(x)
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // z /= log(x)
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

// 10. Ax.
//
// Evaluate a matrix-vector multiplication expression of the form `A * x`,
// where `A` is a floating-point, dense matrix and `x` is a floating-point,
// dense vector.
template<typename E>
struct evaluator<E, typename std::enable_if<is_Ax<E>::value,
                                            void>::type> {
  using value_type = typename E::value_type;

  // y = Ax.
  inline static void assign(const E& expr, value_type* buffer) {
    // TODO(Linh): Do we really need to fill buffer with all zeros first?
    std::fill(buffer, buffer + expr.size(), value_type()/*zero*/);
    internal::insight_gemv(CblasNoTrans,
                           expr.m.num_rows(),
                           expr.m.num_cols(),
                           value_type(1.0),
                           expr.m.begin(),
                           expr.v.begin(),
                           value_type()/*zero*/,
                           buffer);
  }

  // y += Ax.
  inline static void add(const E& expr, value_type* buffer) {
    internal::insight_gemv(CblasNoTrans,
                           expr.m.num_rows(),
                           expr.m.num_cols(),
                           value_type(1.0),
                           expr.m.begin(),
                           expr.v.begin(),
                           value_type(1.0),
                           buffer);
  }

  // y -= Ax
  inline static void sub(const E& expr, value_type* buffer) {
    internal::insight_gemv(CblasNoTrans,
                           expr.m.num_rows(),
                           expr.m.num_cols(),
                           value_type(-1.0),
                           expr.m.begin(),
                           expr.v.begin(),
                           value_type(1.0),
                           buffer);
  }

  // y *= Ax
  inline static void mul(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ *= e; });
  }

  // y /= Ax
  inline static void div(const E& expr, value_type* buffer) {
    std::for_each(expr.begin(), expr.end(),
                  [&](const value_type& e) { *buffer++ /= e; });
  }
};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_EVALUATOR_H_
