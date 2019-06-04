// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_SPECIAL_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_SPECIAL_EXPRESSION_H_

#include "insight/linalg/type_traits/special_dense_expression_traits.h"

namespace insight {

template<typename E> struct is_special_expression
    : public std::conditional<is_special_dense_expression<E>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename E> struct is_special_expression<const E>
    : public is_special_expression<E>{};
template<typename E> struct is_special_expression<volatile E>
    : public is_special_expression<E>{};
template<typename E> struct is_special_expression<volatile const E>
    : public is_special_expression<E>{};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_SPECIAL_EXPRESSION_H_
