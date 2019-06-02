// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_H_

#include "insight/linalg/type_traits/is_fd_times_scalar.h"
#include "insight/linalg/type_traits/is_fd_div_scalar.h"
#include "insight/linalg/type_traits/is_fd_elemwise_op_fd.h"
#include "insight/linalg/type_traits/is_unary_functor_of_fd.h"

namespace insight {

// Type traits for a special binary vector expression.

template<typename E>
struct is_special_binary_expression : public std::false_type{};

template<typename E>
struct is_special_binary_expression<const E>
    : public is_special_binary_expression<E>{};

template<typename E>
struct is_special_binary_expression<volatile const E>
    : public is_special_binary_expression<E>{};

template<typename E>
struct is_special_binary_expression<volatile E>
    : public is_special_binary_expression<E>{};

template<typename E1, typename E2, typename F>
struct is_special_binary_expression< binary_expression<E1, E2, F> >
    : public std::conditional<
  is_fd_times_scalar<binary_expression<E1, E2, F> >::value ||
  is_fd_div_scalar<binary_expression<E1, E2, F> >::value ||
  is_fd_elemwise_op_fd<binary_expression<E1, E2, F> >::value,
  std::true_type,
  std::false_type>::type{};

template<typename E>
struct is_special_unary_expression : public std::false_type{};

template<typename E>
struct is_special_unary_expression<const E>
    : public is_special_binary_expression<E>{};

template<typename E>
struct is_special_unary_expression<volatile const E>
    : public is_special_binary_expression<E>{};

template<typename E>
struct is_special_unary_expression<volatile E>
    : public is_special_binary_expression<E>{};

template<typename E, typename F>
struct is_special_unary_expression< unary_expression<E, F> >
    : public std::conditional<
  is_unary_functor_of_fd<unary_expression<E, F> >::value,
  std::true_type,
  std::false_type>::type{};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_H_
