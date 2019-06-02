// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_MATRIX_MUL_FD_VECTOR_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_MATRIX_MUL_FD_VECTOR_H_

#include "insight/linalg/matrix_mul_vector.h"

namespace insight {

// Is a particular matrix-vector multilication expression a multiplication
// between a floating-point, dense matrix and a floating-point, dense
// vector?

template<typename E> struct is_fd_matrix_mul_fd_vector
    : public std::false_type{};

template<typename E>
struct is_fd_matrix_mul_fd_vector<const E>
    : public is_fd_matrix_mul_fd_vector<E>{};

template<typename E>
struct is_fd_matrix_mul_fd_vector<volatile const E>
    : public is_fd_matrix_mul_fd_vector<E>{};

template<typename E>
struct is_fd_matrix_mul_fd_vector<volatile E>
    : public is_fd_matrix_mul_fd_vector<E>{};

template<typename T, typename A> class vector;
template<typename T, typename A> class matrix;

template<typename T, typename A>
struct is_fd_matrix_mul_fd_vector<matrix_mul_vector<matrix<T, A>,
                                                    vector<T, A> > >
    : public std::conditional<std::is_floating_point<T>::value,
                              std::true_type,
                              std::false_type>::type{};
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_FD_MATRIX_MUL_FD_VECTOR_H_
