// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_H_

#include "insight/linalg/type_traits/is_fd_times_scalar.h"
#include "insight/linalg/type_traits/is_fd_div_scalar.h"
#include "insight/linalg/type_traits/is_fd_plus_scalar.h"
#include "insight/linalg/type_traits/is_fd_minus_scalar.h"
#include "insight/linalg/type_traits/is_scalar_minus_fd.h"
#include "insight/linalg/type_traits/is_alpha_fd_plus_beta.h"
#include "insight/linalg/type_traits/is_alpha_fd_minus_beta.h"
#include "insight/linalg/type_traits/is_beta_minus_alpha_fd.h"

namespace insight {

// Type traits for a normal binary matrix expression.

template<typename E>
struct is_normal_bin_expr
    : public std::conditional<(is_fd_times_scalar<E>::value ||
                               is_fd_div_scalar<E>::value ||
                               is_fd_plus_scalar<E>::value ||
                               is_fd_minus_scalar<E>::value ||
                               is_scalar_minus_fd<E>::value ||
                               is_alpha_fd_plus_beta<E>::value ||
                               is_alpha_fd_minus_beta<E>::value ||
                               is_beta_minus_alpha_fd<E>::value),
                              std::false_type, std::true_type
                              >::type{};

template<typename E>
struct is_normal_bin_expr<const E> : public is_normal_bin_expr<E>{};

template<typename E>
struct is_normal_bin_expr<volatile const E> : public is_normal_bin_expr<E>{};

template<typename E>
struct is_normal_bin_expr<volatile E> : public is_normal_bin_expr<E>{};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_H_
