// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_TRAITS_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_TRAITS_H_

#include "insight/linalg/detail/arithmetic_expression.h"
#include "insight/linalg/detail/transpose_expression.h"
#include "insight/linalg/detail/is_dense_vector.h"
#include "insight/linalg/detail/is_dense_matrix.h"

namespace insight {
namespace linalg_detail {

namespace expression_category {
struct normal{};
struct ax{};      // a * x
struct xpy{};     // x + y
struct xmy{};     // x - y
struct xty{};     // x * y
struct xdy{};     // x / y
};

template<typename E> struct expression_traits {
  using category = expression_category::normal;
};

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
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_TRAITS_H_
