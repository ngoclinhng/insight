// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_DIV_SCALAR_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_DIV_SCALAR_H_

#include "insight/linalg/arithmetic_expression.h"

namespace insight {

// Is a particular binary matrix expression a division of a floating
// point, dense matrix and a floating-point scalar?

template<typename E> struct is_fd_div_scalar : public std::false_type{};

template<typename E>
struct is_fd_div_scalar<const E> : public is_fd_div_scalar<E>{};

template<typename E>
struct is_fd_div_scalar<volatile const E> : public is_fd_div_scalar<E>{};

template<typename E>
struct is_fd_div_scalar<volatile E> : public is_fd_div_scalar<E>{};

template<typename T, typename A> class vector;

template<typename T, typename A>
struct is_fd_div_scalar<binary_expression<vector<T, A>, T, std::divides<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename T, typename A> class matrix;

template<typename T, typename A>
struct is_fd_div_scalar<binary_expression<matrix<T, A>, T, std::divides<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};
}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_DIV_SCALAR_H_
