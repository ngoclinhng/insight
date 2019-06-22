// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_IS_DENSE_VECTOR_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_IS_DENSE_VECTOR_H_

#include <type_traits>

namespace insight {

// These forward declarations should be here NOT inside the linalg_detail
// namespace.
template<typename T, typename A> class vector;
template<typename T, typename A> class matrix;

// Is E a dense (column) vector but not a (column) vector expression?

namespace linalg_detail {

template<typename E> struct is_dense_vector: public std::false_type{};

template<typename E> struct is_dense_vector<const E>
    : public is_dense_vector<E>{};

template<typename E> struct is_dense_vector<volatile E>
    : public is_dense_vector<E>{};

template<typename E> struct is_dense_vector<volatile const E>
    : public is_dense_vector<E>{};

template<typename T, typename A>
struct is_dense_vector<insight::vector<T, A> >
    : public std::true_type{};


// These forward declarations should be here, right inside the linalg_detail
// namespace.
template<typename E> struct row_view;
template<typename E> struct transpose_expression;

template<typename T, typename A>
struct is_dense_vector<
  transpose_expression<row_view<insight::matrix<T, A> > > >
    : public std::true_type{};

}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_IS_DENSE_VECTOR_H_
