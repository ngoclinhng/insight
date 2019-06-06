// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_ROW_VIEW_H_
#define INCLUDE_INSIGHT_LINALG_ROW_VIEW_H_

#include <algorithm>
#include <iterator>

#include "glog/logging.h"

namespace insight {

// Forward declaration for matrix expression.
template<typename Derived> class matrix_expression;

// Forwadr declaration for transpose expression.
template<typename E> class transpose_expression;

// Row view of a generic matrix expression.
// TODO(Linh): Should we restrict `M` to be only matrix expression but not
// vector expression.
template<typename M>
struct row_view : public matrix_expression<row_view<M> > {
 private:
  using self_type = row_view;

 public:
  // Public types.
  using value_type = typename M::value_type;
  using size_type = typename M::size_type;
  using difference_type = typename M::difference_type;
  using reference = typename M::reference;
  using const_reference = typename M::const_reference;
  using pointer = typename M::pointer;
  using const_pointer = typename M::const_pointer;
  using shape_type = typename M::shape_type;

  // TODO(Linh): Should we change this to be true, since row_view can be
  // mathematically viewd as a row vector?
  static constexpr bool is_vector = false;

  row_view(M* m, size_type row_index)
      : row_index_(row_index),
        num_cols_(m->num_cols()),
        begin_(std::next(m->begin(), row_index * num_cols_)),
        end_(std::next(m->begin(), (row_index + 1) * num_cols_)),
        cbegin_(std::next(m->cbegin(), row_index * num_cols_)),
        cend_(std::next(m->cbegin(), (row_index + 1) * num_cols_)) {
    CHECK_LT(row_index, m->num_rows()) << "invalid row index";
  }

  inline size_type num_rows() const { return 1; }
  inline size_type num_cols() const { return num_cols_; }
  inline size_type size() const { return num_rows() * num_cols(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }

  // Transpose of this expression.
  inline transpose_expression<self_type> t() const {
    return transpose_expression<self_type>(*this);
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
  // TODO(Linh): Only anable these arithmetic operations when `M` is
  // of type matrix not matrix expression.

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
