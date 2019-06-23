// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_TRAITS_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_TRAITS_H_

#include "insight/linalg/detail/arithmetic_expression.h"
#include "insight/linalg/detail/transpose_expression.h"
#include "insight/linalg/detail/is_dense_vector.h"
#include "insight/linalg/detail/is_dense_matrix.h"
#include "insight/linalg/detail/functors.h"

namespace insight {
namespace linalg_detail {

// Specific expressions for floating-point, dense matrix arithmetic.
// These are the kinds of expressions that use (typically) BLAS kernels
// to handle.
namespace expression_category {
struct normal{};  // NOT specific.
struct ax{};      // a * x
struct xpy{};     // x + y
struct xmy{};     // x - y
struct xty{};     // x * y
struct xdy{};     // x / y
struct sqrt{};    // sqrt(x)
struct exp{};     // exp(x)
struct log{};     // log(x)
}  // namespace expression_category

// normal expression
template<typename E> struct expression_traits {
  using category = expression_category::normal;
};

// a * x: multiplication between a floating-point, dense matrix/vector (x)
//        with a floating-point, dense scalar (a).

template<typename E, typename T>
struct expression_traits<binary_expression<E, T, std::multiplies<T> > > {
  using category =
      typename std::conditional<(std::is_floating_point<T>::value &&
                                 (is_dense_vector<E>::value ||
                                  is_dense_matrix<E>::value)),
                                expression_category::ax,
                                expression_category::normal>::type;
};

template<typename E, typename T>
struct expression_traits<binary_expression<T, E, std::multiplies<T> > > {
  using category =
      typename std::conditional<(std::is_floating_point<T>::value &&
                                 (is_dense_vector<E>::value ||
                                  is_dense_matrix<E>::value)),
                                expression_category::ax,
                                expression_category::normal>::type;
};

// x + y: addition between two floating-point, dense matrices/vectors.
template<typename E1, typename E2, typename T>
struct expression_traits<binary_expression<E1, E2, std::plus<T> > > {
  using category =
      typename std::conditional<(std::is_floating_point<T>::value &&
                                 ((is_dense_vector<E1>::value &&
                                   is_dense_vector<E2>::value) ||
                                  (is_dense_matrix<E1>::value &&
                                   is_dense_matrix<E2>::value))),
                                expression_category::xpy,
                                expression_category::normal>::type;
};

// x - y: substraction between two floating-point, dense matrices/vectors.
template<typename E1, typename E2, typename T>
struct expression_traits<binary_expression<E1, E2, std::minus<T> > > {
  using category =
      typename std::conditional<(std::is_floating_point<T>::value &&
                                 ((is_dense_vector<E1>::value &&
                                   is_dense_vector<E2>::value) ||
                                  (is_dense_matrix<E1>::value &&
                                   is_dense_matrix<E2>::value))),
                                expression_category::xmy,
                                expression_category::normal>::type;
};

// x * y: element-wise multiplication between two floating-point,
//        dense matrices/vectors.
template<typename E1, typename E2, typename T>
struct expression_traits<binary_expression<E1, E2, std::multiplies<T> > > {
  using category =
      typename std::conditional<(std::is_floating_point<T>::value &&
                                 ((is_dense_vector<E1>::value &&
                                   is_dense_vector<E2>::value) ||
                                  (is_dense_matrix<E1>::value &&
                                   is_dense_matrix<E2>::value))),
                                expression_category::xty,
                                expression_category::normal>::type;
};

// x / y: element-wise division between two floating-point,
//        dense matrices/vectors.
template<typename E1, typename E2, typename T>
struct expression_traits<binary_expression<E1, E2, std::divides<T> > > {
  using category =
      typename std::conditional<(std::is_floating_point<T>::value &&
                                 ((is_dense_vector<E1>::value &&
                                   is_dense_vector<E2>::value) ||
                                  (is_dense_matrix<E1>::value &&
                                   is_dense_matrix<E2>::value))),
                                expression_category::xdy,
                                expression_category::normal>::type;
};

// sqrt(x): element-wise sqrt of a floating-point, dense matrix/vector.
template<typename E>
struct expression_traits<unary_expression<E, sqrt<typename E::value_type> > > {
  using category =
      typename std::conditional<
    (std::is_floating_point<typename E::value_type>::value) &&
    (is_dense_matrix<E>::value || is_dense_vector<E>::value),
    expression_category::sqrt,
    expression_category::normal>::type;
};

// exp(x): element-wise exp of a floating-point, dense matrix/vector.
template<typename E>
struct expression_traits<unary_expression<E, exp<typename E::value_type> > > {
  using category =
      typename std::conditional<
    (std::is_floating_point<typename E::value_type>::value) &&
    (is_dense_matrix<E>::value || is_dense_vector<E>::value),
    expression_category::exp,
    expression_category::normal>::type;
};

// log(x): element-wise log2 of a floating-point, dense matrix/vector.
template<typename E>
struct expression_traits<unary_expression<E, log<typename E::value_type> > > {
  using category =
      typename std::conditional<
    (std::is_floating_point<typename E::value_type>::value) &&
    (is_dense_matrix<E>::value || is_dense_vector<E>::value),
    expression_category::log,
    expression_category::normal>::type;
};
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_TRAITS_H_
