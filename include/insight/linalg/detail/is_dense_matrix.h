// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_IS_DENSE_MATRIX_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_IS_DENSE_MATRIX_H_

#include <type_traits>

namespace insight {

// Forward declaration of the matrix class.
template<typename T, typename A> class matrix;

namespace linalg_detail {

// Is E a dense matrix but not a matrix expression?

template<typename E> struct is_dense_matrix: public std::false_type{};
template<typename E> struct is_dense_matrix<const E>
    : public is_dense_matrix<E>{};
template<typename E> struct is_dense_matrix<volatile E>
    : public is_dense_matrix<E>{};
template<typename E> struct is_dense_matrix<volatile const E>
    : public is_dense_matrix<E>{};

template<typename T, typename A>
struct is_dense_matrix<insight::matrix<T, A> >: public std::true_type{};

// Since we have only one kind of vector which is column vector, so
// a particular row of a dense matrix is also a dense matrix.
template<typename E> struct row_view;
template<typename T, typename A>
struct is_dense_matrix<
  row_view<insight::matrix<T, A> > > : public std::true_type{};


}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_IS_DENSE_MATRIX_H_
