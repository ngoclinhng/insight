// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_ELEMWISE_OP_FD_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_ELEMWISE_OP_FD_H_

#include "insight/linalg/arithmetic_expression.h"

namespace insight {

// Is a particular binary expression an element-wise operation between
// two dense, floating-point vectors/matrices.

template<typename E> struct is_fd_elemwise_op_fd : public std::false_type{};

template<typename E>
struct is_fd_elemwise_op_fd<const E> : public is_fd_elemwise_op_fd<E>{};

template<typename E>
struct is_fd_elemwise_op_fd<volatile const E>
    : public is_fd_elemwise_op_fd<E>{};

template<typename E>
struct is_fd_elemwise_op_fd<volatile E> : public is_fd_elemwise_op_fd<E>{};

template<typename T, typename A> class vector;

template<typename T, typename A, typename F>
struct is_fd_elemwise_op_fd<binary_expression<vector<T, A>, vector<T, A>, F> >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};
}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_ELEMWISE_OP_FD_H_
