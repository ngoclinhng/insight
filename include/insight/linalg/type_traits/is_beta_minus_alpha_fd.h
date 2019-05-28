// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_BETA_MINUS_ALPHA_FD_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_BETA_MINUS_ALPHA_FD_H_

#include "insight/linalg/type_traits/is_fd_times_scalar.h"

namespace insight {
// Is a particular binary matrix expresson of the form: β - αA, where A
// is a floating-point, dense matrix, and alpha, beta are floating-point
// scalars?

template<typename E> struct is_beta_minus_alpha_fd : public std::false_type{};

template<typename E>
struct is_beta_minus_alpha_fd<const E> : public is_beta_minus_alpha_fd<E>{};

template<typename E>
struct is_beta_minus_alpha_fd<volatile const E> : public is_beta_minus_alpha_fd<E>{};  // NOLINT

template<typename E>
struct is_beta_minus_alpha_fd<volatile E> : public is_beta_minus_alpha_fd<E>{};

template<typename E, typename T>
struct is_beta_minus_alpha_fd<binary_expr<T, E, std::minus<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               is_fd_times_scalar<E>::value),
                              std::true_type,
                              std::false_type>::type{};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_BETA_MINUS_ALPHA_FD_H_
