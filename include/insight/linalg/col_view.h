// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_COL_VIEW_H_
#define INCLUDE_INSIGHT_LINALG_COL_VIEW_H_

#include <algorithm>
#include <iterator>

#include "glog/logging.h"

namespace insight {

// Forward declaration for vector expression.
template<typename Derived> struct vector_expression;

// Forward declaration for transpose expression.
template<typename E> struct transpose_expression;

// Row view of a row.
// TODO(Linh): Should we restrict `M` to be only matrix expression but not
// vector expression.
template<typename M>
struct col_view : public vector_expression<col_view<M> > {
 private:
  using self_type = col_view;

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

  static constexpr bool is_vector = true;

  col_view(M* m, size_type col_index) : m_(m), col_index_(col_index) {
    CHECK_LT(col_index, m->num_cols()) << "Invalid column index";
  }

  inline size_type num_rows() const { return m_->num_rows(); }
  inline size_type num_cols() const { return 1; }
  inline size_type size() const { return num_rows() * num_cols(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }

  // Transpose of this expression.
  inline transpose_expression<self_type> t() const {
    return transpose_expression<self_type>(*this);
  }

  class const_iterator;

  // Iterator.
  class iterator {
   private:
    using subiterator_type = typename M::iterator;

   public:
    // public types.
    using value_type = typename col_view::value_type;
    using difference_type = typename col_view::difference_type;
    using pointer = typename col_view::pointer;
    using reference = typename col_view::reference;
    using iterator_category = std::input_iterator_tag;

    iterator() : row_index_(), row_size_(), it_() {}

    iterator(col_view* col, size_type row_index)
        : row_index_(row_index),
          row_size_(col->m_->num_cols()),
          it_(std::next(col->m_->begin(), col->col_index_)) {}

    // Copy constructor.
    iterator(const iterator& it)
        : row_index_(it.row_index_),
          row_size_(it.row_size_),
          it_(it.it_) {}

    // Assignment operator.
    iterator& operator=(const iterator& it) {
      if (this == &it) { return *this; }
      row_index_ = it.row_index_;
      row_size_ = it.row_size_;
      it_ = it.it_;
      return *this;
    }

    // Dereference.
    inline reference operator*() {
      return *it_;
    }

    // Comparison.

    inline bool operator==(const iterator& it) const {
      return (row_index_ == it.row_index_);
    }

    inline bool operator!=(const iterator& it) const {
      return !(*this == it);
    }

    inline bool operator==(const const_iterator& it) const {
      return (row_index_ == it.row_index_);
    }

    inline bool operator!=(const const_iterator& it) const {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline iterator& operator++() {
      ++row_index_;
      std::advance(it_, row_size_);
      return *this;
    }

    // Postfix increment it++.
    inline iterator operator++(int) {
      iterator temp(*this);
      ++(*this);
      return temp;
    }

    friend class const_iterator;

   private:
    size_type row_index_;
    size_type row_size_;
    subiterator_type it_;
  };

  // Const Iterator.
  class const_iterator {
   private:
    using const_subiterator_type = typename M::const_iterator;

   public:
    // public types.
    using value_type = typename col_view::value_type;
    using difference_type = typename col_view::difference_type;
    using pointer = typename col_view::const_pointer;
    using reference = typename col_view::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : row_index_(), row_size_(), it_() {}

    const_iterator(const col_view& col, size_type row_index)
        : row_index_(row_index),
          row_size_(col.m_->num_cols()),
          it_(std::next(col.m_->cbegin(), col.col_index_)) {}

    // Copy constructor.
    const_iterator(const const_iterator& it)
        : row_index_(it.row_index_),
          row_size_(it.row_size_),
          it_(it.it_) {}

    // Conversion from iterator to const_iterator.
    const_iterator(const iterator& it)  // NOLINT
        : row_index_(it.row_index_),
          row_size_(it.row_size_),
          it_(it.it_) {}

    // Assignment operator.
    const_iterator& operator=(const const_iterator& it) {
      if (this == &it) { return *this; }
      row_index_ = it.row_index_;
      row_size_ = it.row_size_;
      it_ = it.it_;
      return *this;
    }

    // Dereference.
    inline reference operator*() const {
      return *it_;
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) const {
      return (row_index_ == it.row_index_);
    }

    inline bool operator!=(const const_iterator& it) const {
      return !(*this == it);
    }

    inline bool operator==(const iterator& it) const {
      return (row_index_ == it.row_index_);
    }

    inline bool operator!=(const iterator& it) const {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline const_iterator& operator++() {
      ++row_index_;
      std::advance(it_, row_size_);
      return *this;
    }

    // Postfix increment it++.
    inline const_iterator operator++(int) {
      const_iterator temp(*this);
      ++(*this);
      return temp;
    }

   private:
    size_type row_index_;
    size_type row_size_;
    const_subiterator_type it_;
  };

  inline iterator begin() { return iterator(this, 0); }
  inline const_iterator begin() const { return const_iterator(*this, 0); }
  inline const_iterator cbegin() const { return const_iterator(*this, 0); }

  inline iterator end() { return iterator(this, num_rows()); }

  inline const_iterator end() const {
    return const_iterator(*this, num_rows());
  }

  inline const_iterator cend() const {
    return const_iterator(*this, num_rows());
  }


  // col_view-scalar arithmetic.
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
  M* m_;
  size_type col_index_;
};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_COL_VIEW_H_
