// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_ROW_VIEW_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_ROW_VIEW_H_

#include <algorithm>
#include <iterator>

#include "glog/logging.h"

namespace insight {
namespace linalg_detail {

// Forward declarations
template<typename Derived> class matrix_expression;
template<typename E> class transpose_expression;

// Row view of a generic matrix expression.
// TODO(Linh): Should we restrict `M` to be only matrix expression but not
// vector expression.
template<typename M>
struct row_view : public matrix_expression<row_view<M> > {
 private:
  using self = row_view<M>;
  using iter_traits = std::iterator_traits<typename M::iterator>;

 public:
  using value_type = typename iter_traits::value_type;
  using reference = typename iter_traits::reference;
  using size_type = typename M::size_type;
  using iterator = typename M::iterator;
  using const_iterator = typename M::const_iterator;
  using shape_type = typename M::shape_type;

  row_view(M* m, size_type row_index)
      : row_index_(row_index),
        col_count_(m->col_count()),
        begin_(std::next(m->begin(), row_index * col_count_)),
        end_(std::next(m->begin(), (row_index + 1) * col_count_)),
        cbegin_(std::next(m->cbegin(), row_index * col_count_)),
        cend_(std::next(m->cbegin(), (row_index + 1) * col_count_)) {
    CHECK_LT(row_index, m->row_count()) << "invalid row index";
  }

  inline size_type row_count() const { return col_count_ > 0 ? 1 : 0; }
  inline size_type col_count() const { return col_count_; }
  inline size_type size() const { return row_count() * col_count(); }
  inline shape_type shape() const {
    return shape_type(row_count(), col_count());
  }

  // Transpose of this expression.
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  inline iterator begin() { return begin_; }
  inline const_iterator begin() const { return cbegin_; }
  inline const_iterator cbegin() const { return cbegin_; }

  inline iterator end() { return end_; }
  inline const_iterator end() const { return cend_; }
  inline const_iterator cend() const { return cend_; }

  // row_view-scalar arithmetic.

  // Increments each and every element in the row by the constant
  // `scalar`.
  inline self& operator+=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e += scalar; });
    return *this;
  }

  // Decrements each and every element in the row by the constant
  // `scalar`.
  inline self& operator-=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e -= scalar; });
    return *this;
  }

  // Replaces each and every element in the row by the result of
  // multiplication of that element and a `scalar`.
  inline self& operator*=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e *= scalar; });
    return *this;
  }

  // Replaces each and every element in the row by the result of
  // dividing that element by a constant `scalar`.
  inline self& operator/=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e /= scalar; });
    return *this;
  }

 private:
  size_type row_index_;
  size_type col_count_;
  iterator begin_;
  iterator end_;
  const_iterator cbegin_;
  const_iterator cend_;
};
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_ROW_VIEW_H_
