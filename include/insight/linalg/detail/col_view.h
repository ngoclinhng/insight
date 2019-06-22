// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_COL_VIEW_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_COL_VIEW_H_

#include <algorithm>
#include <iterator>

#include "insight/internal/jump_iterator.h"
#include "glog/logging.h"

namespace insight {
namespace linalg_detail {

// Forward declarations
template<typename Derived> struct vector_expression;
template<typename E> struct transpose_expression;

// Column view of a matrix/matrix_expression.
template<typename M>
struct col_view : public vector_expression<col_view<M> > {
 private:
  using self = col_view;
  using iter_traits = std::iterator_traits<typename M::iterator>;

 public:
  using value_type = typename iter_traits::value_type;
  using reference = typename iter_traits::reference;
  using size_type = typename M::size_type;
  using iterator = internal::jump_iterator<typename M::iterator>;
  using const_iterator = internal::jump_iterator<typename M::const_iterator>;
  using shape_type = typename M::shape_type;

  col_view(M* m, size_type col_index)
      : begin_(std::next(m->begin(), col_index)),
        end_(m->end()),
        cbegin_(std::next(m->cbegin(), col_index)),
        cend_(m->cend()),
        row_count_(m->row_count()),
        col_count_(m->col_count()) {
    CHECK_LT(col_index, m->col_count()) << "Invalid column index";
  }

  inline size_type row_count() const { return row_count_; }
  inline size_type col_count() const { return row_count_ > 0 ? 1 : 0; }
  inline size_type size() const { return row_count() * col_count(); }
  inline shape_type shape() const {
    return shape_type(row_count(), col_count());
  }

  // Transpose of this expression.
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }


  inline iterator begin() {
    return internal::make_jump_iterator(begin_, col_count_, 0,
                                        std::distance(begin_, end_));
  }

  inline const_iterator begin() const {
    return internal::make_jump_iterator(cbegin_, col_count_, 0,
                                        std::distance(cbegin_, cend_));
  }

  inline const_iterator cbegin() const {
    return internal::make_jump_iterator(cbegin_, col_count_, 0,
                                        std::distance(cbegin_, cend_));
  }

  inline iterator end() {
    return internal::make_jump_iterator(end_, col_count_,
                                        std::distance(begin_, end_), 0);
  }

  inline const_iterator end() const {
    return internal::make_jump_iterator(cend_, col_count_,
                                        std::distance(cbegin_, cend_), 0);
  }

  inline const_iterator cend() const {
    return internal::make_jump_iterator(cend_, col_count_,
                                        std::distance(cbegin_, cend_), 0);
  }


  // col_view-scalar arithmetic.
  // TODO(Linh): Only anable these arithmetic operations when `M` is
  // of type matrix not matrix expression.

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
  typename M::iterator begin_;
  typename M::iterator end_;
  typename M::const_iterator cbegin_;
  typename M::const_iterator cend_;
  size_type row_count_;
  size_type col_count_;
};

}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_COL_VIEW_H_
