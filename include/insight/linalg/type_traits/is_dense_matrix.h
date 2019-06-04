// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_DENSE_MATRIX_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_DENSE_MATRIX_H_

#include <type_traits>

namespace insight {

// Is E a dense matrix but not a matrix expression?

// Forward declaration of the matrix class.
template<typename T, typename A> class matrix;

template<typename E> struct is_dense_matrix: public std::false_type{};
template<typename E> struct is_dense_matrix<const E>
    : public is_dense_matrix<E>{};
template<typename E> struct is_dense_matrix<volatile E>
    : public is_dense_matrix<E>{};
template<typename E> struct is_dense_matrix<volatile const E>
    : public is_dense_matrix<E>{};

template<typename T, typename A>
struct is_dense_matrix<matrix<T, A> >: public std::true_type{};
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_DENSE_MATRIX_H_
