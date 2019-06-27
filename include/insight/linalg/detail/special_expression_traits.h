// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_TRAITS_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_TRAITS_H_

#include "insight/linalg/detail/arithmetic_expression.h"
#include "insight/linalg/detail/transpose_expression.h"
#include "insight/linalg/detail/matmul_expression.h"
#include "insight/linalg/detail/is_dense_vector.h"
#include "insight/linalg/detail/is_dense_matrix.h"
#include "insight/linalg/detail/functors.h"

namespace insight {
namespace linalg_detail {
namespace special_expression {

// ax.

template<typename E> struct is_ax : public std::false_type{};

template<typename E, typename T>
struct is_ax<binary_expression<E, T, std::multiplies<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              (is_dense_vector<E>::value ||
                               is_dense_matrix<E>::value),
                              std::true_type,
                              std::false_type>::type{};

template<typename E, typename T>
struct is_ax<binary_expression<T, E, std::multiplies<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              (is_dense_vector<E>::value ||
                               is_dense_matrix<E>::value),
                              std::true_type,
                              std::false_type>::type{};

// x + y.

template<typename E> struct is_xpy : public std::false_type{};

template<typename E1, typename E2, typename T>
struct is_xpy<binary_expression<E1, E2, std::plus<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              ((is_dense_vector<E1>::value &&
                                is_dense_vector<E2>::value) ||
                               (is_dense_matrix<E1>::value &&
                                is_dense_matrix<E2>::value)),
                              std::true_type,
                              std::false_type>::type{};

// x - y.

template<typename E> struct is_xmy : public std::false_type{};

template<typename E1, typename E2, typename T>
struct is_xmy<binary_expression<E1, E2, std::minus<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              ((is_dense_vector<E1>::value &&
                                is_dense_vector<E2>::value) ||
                               (is_dense_matrix<E1>::value &&
                                is_dense_matrix<E2>::value)),
                              std::true_type,
                              std::false_type>::type{};

// x * y.

template<typename E> struct is_xty : public std::false_type{};

template<typename E1, typename E2, typename T>
struct is_xty<binary_expression<E1, E2, std::multiplies<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              ((is_dense_vector<E1>::value &&
                                is_dense_vector<E2>::value) ||
                               (is_dense_matrix<E1>::value &&
                                is_dense_matrix<E2>::value)),
                              std::true_type,
                              std::false_type>::type{};

// x / y.

template<typename E> struct is_xdy : public std::false_type{};

template<typename E1, typename E2, typename T>
struct is_xdy<binary_expression<E1, E2, std::divides<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              ((is_dense_vector<E1>::value &&
                                is_dense_vector<E2>::value) ||
                               (is_dense_matrix<E1>::value &&
                                is_dense_matrix<E2>::value)),
                              std::true_type,
                              std::false_type>::type{};

// sqrt(x).

template<typename E> struct is_sqrt_of_x : public std::false_type{};

template<typename E>
struct is_sqrt_of_x<unary_expression<E, sqrt<typename E::value_type> > >
    : public std::conditional<
  std::is_floating_point<typename E::value_type>::value &&
  (is_dense_vector<E>::value || is_dense_matrix<E>::value),
  std::true_type,
  std::false_type>::type{};

// exp(x).

template<typename E> struct is_exp_of_x : public std::false_type{};

template<typename E>
struct is_exp_of_x<unary_expression<E, exp<typename E::value_type> > >
    : public std::conditional<
  std::is_floating_point<typename E::value_type>::value &&
  (is_dense_vector<E>::value || is_dense_matrix<E>::value),
  std::true_type,
  std::false_type>::type{};

// log(x).

template<typename E> struct is_log_of_x : public std::false_type{};

template<typename E>
struct is_log_of_x<unary_expression<E, log<typename E::value_type> > >
    : public std::conditional<
  std::is_floating_point<typename E::value_type>::value &&
  (is_dense_vector<E>::value || is_dense_matrix<E>::value),
  std::true_type,
  std::false_type>::type{};

// Is a generic expression E of the form a * x or x * a where a is a scalar,
// and x is a dense vector having the same element type as a?

template<typename E>
struct is_dense_vector_times_scalar : public std::false_type{};

template<typename E, typename T>
struct is_dense_vector_times_scalar<
  binary_expression<E, T, std::multiplies<T> > >
    : public std::conditional<
  is_dense_vector<E>::value &&
  std::is_same<typename E::value_type, T>::value,
  std::true_type,
  std::false_type>::type{};

template<typename E, typename T>
struct is_dense_vector_times_scalar<
  binary_expression<T, E, std::multiplies<T> > >
    : public std::conditional<
  is_dense_vector<E>::value &&
  std::is_same<typename E::value_type, T>::value,
  std::true_type,
  std::false_type>::type{};

// Is a generic expression E of the form a * A or A * a where a is a scalar,
// and A is a dense matrix having the same element type as a?

template<typename E>
struct is_dense_matrix_times_scalar : public std::false_type{};

template<typename E, typename T>
struct is_dense_matrix_times_scalar<
  binary_expression<E, T, std::multiplies<T> > >
    : public std::conditional<
  is_dense_matrix<E>::value &&
  std::is_same<typename E::value_type, T>::value,
  std::true_type,
  std::false_type>::type{};

template<typename E, typename T>
struct is_dense_matrix_times_scalar<
  binary_expression<T, E, std::multiplies<T> > >
    : public std::conditional<
  is_dense_matrix<E>::value &&
  std::is_same<typename E::value_type, T>::value,
  std::true_type,
  std::false_type>::type{};

// Is a generic expression E of the form A' (transpose of A) where
// A is a dense matrix?

template<typename E>
struct is_transpose_of_dense_matrix : public std::false_type{};

template<typename E>
struct is_transpose_of_dense_matrix<transpose_expression<E> >
    : public std::conditional<is_dense_matrix<E>::value,
                              std::true_type,
                              std::false_type>::type{};

// Is a generic expression of the form a * A' or A' * a where
// a is a scalar, and A is a dense matrix having the same element type
// as a?

template<typename E>
struct is_transpose_of_dense_matrix_times_scalar
    : public std::false_type{};

template<typename E, typename T>
struct is_transpose_of_dense_matrix_times_scalar<
  binary_expression<transpose_expression<E>, T, std::multiplies<T> > >
    : public std::conditional<
  is_dense_matrix<E>::value &&
  std::is_same<typename E::value_type, T>::value,
  std::true_type,
  std::false_type>{};

template<typename E, typename T>
struct is_transpose_of_dense_matrix_times_scalar<
  binary_expression<T, transpose_expression<E>, std::multiplies<T> > >
    : public std::conditional<
  is_dense_matrix<E>::value &&
  std::is_same<typename E::value_type, T>::value,
  std::true_type,
  std::false_type>{};

// matmul(aA, bx): is_matmul_aAbx

template<typename E> struct is_matmul_aAbx : public std::false_type{};

template<typename M, typename V>
struct is_matmul_aAbx<matmul_expression<M, V> >
    : public std::conditional<
  (is_dense_matrix<M>::value || is_dense_matrix_times_scalar<M>::value) &&
  (is_dense_vector<V>::value || is_dense_vector_times_scalar<V>::value) &&
  std::is_same<typename M::value_type, typename V::value_type>::value &&
  std::is_floating_point<typename M::value_type>::value,
  std::true_type,
  std::false_type>::type{};

// helper for evaluating the matmul(aA, bx) expression.
template<typename M, typename V>
struct matmul_aAbx_wrapper {
  using expression_type = matmul_expression<M, V>;
  using value_type = typename expression_type::value_type;
  using matrix_size_type = typename M::size_type;

  const expression_type& expr;

  explicit matmul_aAbx_wrapper(const expression_type& expr,
                       typename std::enable_if<
                       is_matmul_aAbx<expression_type>::value>::type* = 0)
      : expr(expr) {}

  inline matrix_size_type A_row_count() const { return expr.m.row_count(); }
  inline matrix_size_type A_col_count() const { return expr.m.col_count(); }

  inline value_type a() const {
    return a_(std::integral_constant<bool,
              is_dense_matrix_times_scalar<M>::value>());
  }

  inline value_type b() const {
    return b_(std::integral_constant<bool,
              is_dense_vector_times_scalar<V>::value>());
  }

  inline const value_type* A() const {
    return A_(std::integral_constant<bool,
              is_dense_matrix_times_scalar<M>::value>());
  }

  inline const value_type* x() const {
    return x_(std::integral_constant<bool,
              is_dense_vector_times_scalar<V>::value>());
  }

 private:
  inline value_type a_(std::true_type) const {
    return expr.m.scalar;
  }

  inline value_type a_(std::false_type) const {
    return value_type(1.0);
  }

  inline value_type b_(std::true_type) const {
    return expr.v.scalar;
  }

  inline value_type b_(std::false_type) const {
    return value_type(1.0);
  }

  inline const value_type* A_(std::true_type) const {
    return expr.m.e.data();
  }

  inline const value_type* A_(std::false_type) const {
    return expr.m.data();
  }

  inline const value_type* x_(std::true_type) const {
    return expr.v.e.data();
  }

  inline const value_type* x_(std::false_type) const {
    return expr.v.data();
  }
};


// matmul(a * A.t(), bx)

template<typename E> struct is_matmul_aAtbx : public std::false_type{};

template<typename M, typename V>
struct is_matmul_aAtbx<matmul_expression<M, V> >
    : public std::conditional<
  (is_transpose_of_dense_matrix<M>::value ||
   is_transpose_of_dense_matrix_times_scalar<M>::value) &&
  (is_dense_vector<V>::value || is_dense_vector_times_scalar<V>::value) &&
  std::is_same<typename M::value_type, typename V::value_type>::value &&
  std::is_floating_point<typename M::value_type>::value,
  std::true_type,
  std::false_type>::type{};

// helper for evaluating the matmul(aA.t(), bx) expression.
template<typename M, typename V>
struct matmul_aAtbx_wrapper {
  using expression_type = matmul_expression<M, V>;
  using value_type = typename expression_type::value_type;
  using matrix_size_type = typename M::size_type;

  const expression_type& expr;

  explicit matmul_aAtbx_wrapper(
      const expression_type& expr,
      typename std::enable_if<
      is_matmul_aAtbx<expression_type>::value>::type* = 0)
      : expr(expr) {}

  inline matrix_size_type A_row_count() const { return expr.m.col_count(); }
  inline matrix_size_type A_col_count() const { return expr.m.row_count(); }

  inline value_type a() const {
    return a_(std::integral_constant<bool,
              is_transpose_of_dense_matrix_times_scalar<M>::value>());
  }

  inline value_type b() const {
    return b_(std::integral_constant<bool,
              is_dense_vector_times_scalar<V>::value>());
  }

  inline const value_type* A() const {
    return A_(std::integral_constant<bool,
              is_transpose_of_dense_matrix_times_scalar<M>::value>());
  }

  inline const value_type* x() const {
    return x_(std::integral_constant<bool,
              is_dense_vector_times_scalar<V>::value>());
  }

 private:
  inline value_type a_(std::true_type) const {
    return expr.m.scalar;
  }

  inline value_type a_(std::false_type) const {
    return value_type(1.0);
  }

  inline value_type b_(std::true_type) const {
    return expr.v.scalar;
  }

  inline value_type b_(std::false_type) const {
    return value_type(1.0);
  }

  inline const value_type* A_(std::true_type) const {
    return expr.m.e.e.data();
  }

  inline const value_type* A_(std::false_type) const {
    return expr.m.e.data();
  }

  inline const value_type* x_(std::true_type) const {
    return expr.v.e.data();
  }

  inline const value_type* x_(std::false_type) const {
    return expr.v.data();
  }
};
}  // namespace special_expression
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_TRAITS_H_
