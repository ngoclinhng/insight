// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_ROW_VIEW_H_
#define INCLUDE_INSIGHT_LINALG_ROW_VIEW_H_

#include <algorithm>
#include <iterator>

namespace insight {

// Forward declaration for row expression.
template<typename Derived> class matrix_expression;

// Row view of a row.
template<typename M>
struct row_view : public matrix_expression<row_view<M> > {
 private:
  using self_type = row_view;

 public:
  // Public types.
  using value_type = typename M::value_type;
  using size_type = typename M::size_type;
  using difference_type = typename M::difference_type;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = typename M::pointer;
  using const_pointer = typename M::const_pointer;
  using shape_type = typename M::shape_type;

  row_view(M* m, size_type row_index)
      : row_index_(row_index),
        num_cols_(m->num_cols()),
        begin_(std::next(m->begin(), row_index * num_cols_)),
        end_(std::next(m->begin(), (row_index + 1) * num_cols_)),
        cbegin_(std::next(m->cbegin(), row_index * num_cols_)),
        cend_(std::next(m->cbegin(), (row_index + 1) * num_cols_)) {
    // TODO(Linh): Check to make sure that row_index is in the range
    // [0, m->num_rows()).
  }

  inline size_type num_rows() const { return 1; }
  inline size_type num_cols() const { return num_cols_; }
  inline size_type size() const { return num_rows() * num_cols(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }

  // Iterators.
  using iterator = typename M::iterator;
  using const_iterator = typename M::const_iterator;

  inline iterator begin() { return begin_; }
  inline const_iterator begin() const { return cbegin_; }
  inline const_iterator cbegin() const { return cbegin_; }

  inline iterator end() { return end_; }
  inline const_iterator end() const { return cend_; }
  inline const_iterator cend() const { return cend_; }

  // row_view-scalar arithmetic.

  // Increments each and every element in the row by the constant
  // `scalar`.
  inline self_type& operator+=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e += scalar; });
    return *this;
  }

  // Decrements each and every element in the row by the constant
  // `scalar`.
  inline self_type& operator-=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e -= scalar; });
    return *this;
  }

  // Replaces each and every element in the row by the result of
  // multiplication of that element and a `scalar`.
  inline self_type& operator*=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e *= scalar; });
    return *this;
  }

  // Replaces each and every element in the row by the result of
  // dividing that element by a constant `scalar`.
  inline self_type& operator/=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e /= scalar; });
    return *this;
  }

 private:
  size_type row_index_;
  size_type num_cols_;
  iterator begin_;
  iterator end_;
  const_iterator cbegin_;
  const_iterator cend_;
};
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_ROW_VIEW_H_
