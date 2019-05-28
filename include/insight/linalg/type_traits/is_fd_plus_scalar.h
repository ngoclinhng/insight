// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_PLUS_SCALAR_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_PLUS_SCALAR_H_

#include "insight/linalg/matrix_expression.h"
#include "insight/linalg/matrix.h"

namespace insight {

// Is a particular binary matrix expression an addition between
// a floaitng-point, dense matrix and a floating-point scalar?

template<typename E> struct is_fd_plus_scalar : public std::false_type{};

template<typename E>
struct is_fd_plus_scalar<const E> : public is_fd_plus_scalar<E>{};

template<typename E>
struct is_fd_plus_scalar<volatile const E> : public is_fd_plus_scalar<E>{};

template<typename E>
struct is_fd_plus_scalar<volatile E> : public is_fd_plus_scalar<E>{};

template<typename T>
struct is_fd_plus_scalar<binary_expr<matrix<T>, T, std::plus<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename T>
struct is_fd_plus_scalar<binary_expr<T, matrix<T>, std::plus<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename T>
struct is_fd_plus_scalar<binary_expr<typename matrix<T>::row_view, T,
                                    std::plus<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename T>
struct is_fd_plus_scalar<binary_expr<T, typename matrix<T>::row_view,
                                    std::plus<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};
}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_PLUS_SCALAR_H_
