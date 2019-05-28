// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_ALPHA_FD_MINUS_BETA_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_ALPHA_FD_MINUS_BETA_H_

#include "insight/linalg/type_traits/is_fd_times_scalar.h"

namespace insight {

// Is a particular binary matrix expression of the form: αA - β, where
// A is a floating-point, dense matrix, and alpha, beta are floating-point
// scalars.

template<typename E> struct is_alpha_fd_minus_beta : public std::false_type{};

template<typename E>
struct is_alpha_fd_minus_beta<const E> : public is_alpha_fd_minus_beta<E>{};

template<typename E>
struct is_alpha_fd_minus_beta<volatile const E> : public is_alpha_fd_minus_beta<E>{};  // NOLINT

template<typename E>
struct is_alpha_fd_minus_beta<volatile E> : public is_alpha_fd_minus_beta<E>{};

template<typename E, typename T>
struct is_alpha_fd_minus_beta<binary_expr<E, T, std::minus<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               is_fd_times_scalar<E>::value),
                              std::true_type,
                              std::false_type>::type{};
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_ALPHA_FD_MINUS_BETA_H_
