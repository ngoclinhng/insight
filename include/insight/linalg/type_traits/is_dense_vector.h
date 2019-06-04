// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_DENSE_VECTOR_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_DENSE_VECTOR_H_

#include <type_traits>

namespace insight {

// Is E a dense vector but not a vector expression?

// Forward declaration of the vector class.
template<typename T, typename A> class vector;

template<typename E> struct is_dense_vector: public std::false_type{};
template<typename E> struct is_dense_vector<const E>
    : public is_dense_vector<E>{};
template<typename E> struct is_dense_vector<volatile E>
    : public is_dense_vector<E>{};
template<typename E> struct is_dense_vector<volatile const E>
    : public is_dense_vector<E>{};

template<typename T, typename A>
struct is_dense_vector<vector<T, A> >: public std::true_type{};  // NOLINT

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_IS_DENSE_VECTOR_H_
