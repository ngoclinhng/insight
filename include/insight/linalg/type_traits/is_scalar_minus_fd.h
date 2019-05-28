// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_SCALAR_MINUS_FD_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_SCALAR_MINUS_FD_H_

#include "insight/linalg/matrix_expression.h"
#include "insight/linalg/matrix.h"

namespace insight {

// Is a particular binary matrix expression a substraction between
// a floating-point scalar and a floating-point, dense matrix?

template<typename E> struct is_scalar_minus_fd : public std::false_type{};

template<typename E>
struct is_scalar_minus_fd<const E> : public is_scalar_minus_fd<E>{};

template<typename E>
struct is_scalar_minus_fd<volatile const E> : public is_scalar_minus_fd<E>{};

template<typename E>
struct is_scalar_minus_fd<volatile E> : public is_scalar_minus_fd<E>{};

template<typename T>
struct is_scalar_minus_fd<binary_expr<T, matrix<T>, std::minus<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename T>
struct is_scalar_minus_fd<binary_expr<T, typename matrix<T>::row_view,
                                      std::minus<T> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};

}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_SCALAR_MINUS_FD_H_
