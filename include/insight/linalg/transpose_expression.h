// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TRANSPOSE_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_TRANSPOSE_EXPRESSION_H_

#include <iterator>

namespace insight {

// Forward declaration of matrix_expression
template<typename Derived> struct matrix_expression;

// Forward declaration of vector_expression
template<typename Derived> struct vector_expression;

// Forward declaration for row_view.
template<typename M> struct row_view;

// Forward declaration for col_view.
template<typename M> struct col_view;

// Transpose a generic matrix/vector expression.
template<typename E>
struct transpose_expression
    : public matrix_expression<transpose_expression<E> > {
 private:
  using self_type = transpose_expression<E>;

 public:
  // public types.
  using value_type = typename E::value_type;
  using size_type = typename E::size_type;
  using difference_type = typename E::difference_type;
  using const_reference = typename E::const_reference;
  using reference = typename E::reference;
  using const_pointer = typename E::const_pointer;
  using pointer = typename E::pointer;
  using shape_type = typename E::shape_type;

  // TODO(Linh): When we transpose a matrix expression we'll get back
  // a matrix expression, so `is_vector = false` makes sense in this case,
  // but what happens when we transpose a (column) vector expression?
  static constexpr bool is_vector = false;

  const E& e;

  explicit transpose_expression(const E& e) : e(e) {}

  inline size_type num_rows() const { return e.num_cols(); }
  inline size_type num_cols() const { return e.num_rows(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }
  inline size_type size() const { return e.size(); }

  inline row_view<self_type> row_at(size_type row_index) {
    return row_view<self_type>(this, row_index);
  }

  inline col_view<self_type> col_at(size_type col_index) {
    return col_view<self_type>(this, col_index);
  }

  // Iterator.

  class const_iterator;
  using iterator = const_iterator;

  class const_iterator {
   private:
    using const_subiterator_type = typename E::const_iterator;

   public:
    // public types.
    using value_type = typename transpose_expression::value_type;
    using difference_type = typename transpose_expression::difference_type;
    using pointer = typename transpose_expression::const_pointer;
    using reference = typename transpose_expression::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : index_(),
                       count_(),
                       num_rows_(),
                       num_cols_(),
                       row_it_(),
                       col_it_() {}

    const_iterator(const transpose_expression& expr, size_type index)
        : index_(index),
          count_(value_type()/*zero*/),
          num_rows_(expr.num_cols()),
          num_cols_(expr.num_rows()),
          row_it_(expr.e.begin()),
          col_it_(expr.e.begin()) {}

    // Copy constructor.
    const_iterator(const const_iterator& it)
        : index_(it.index_),
          count_(it.count_),
          num_rows_(it.num_rows_),
          num_cols_(it.num_cols_),
          row_it_(it.row_it_),
          col_it_(it.col_it_) {}

    // Assignment operator.
    const_iterator& operator=(const const_iterator& it) {
      if (this == &it) { return *this; }
      index_ = it.index_;
      count_ = it.count_;
      num_rows_ = it.num_rows_;
      num_cols_ = it.num_cols_;
      row_it_ = it.row_it_;
      col_it_ = it.col_it_;
      return *this;
    }

    // Dereference.
    inline reference operator*() const {
      return *row_it_;
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) const {
      return (index_ == it.index_);
    }

    inline bool operator!=(const const_iterator& it) const {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline const_iterator& operator++() {
      ++count_;

      if (count_ == num_rows_) {
        count_ = 0;
        ++col_it_;
        row_it_ = col_it_;
      } else {
        std::advance(row_it_, num_cols_);
      }
      ++index_;
      return *this;
    }

    // Postfix increment it++.
    inline const_iterator operator++(int) {
      const_iterator temp(*this);
      ++(*this);
      return temp;
    }

   private:
    size_type index_;
    size_type count_;
    size_type num_rows_;
    size_type num_cols_;
    const_subiterator_type row_it_;
    const_subiterator_type col_it_;
  };

  inline const_iterator begin() const { return const_iterator(*this, 0); }
  inline const_iterator cbegin() const { return begin(); }
  inline const_iterator end() const { return const_iterator(*this, size()); }
  inline const_iterator cend() const  {return end(); }
};

// Forward declaration of matrix class.
template<typename T, typename A> class matrix;

// Transpose of a row_view.
// TODO(Linh): Give some explanation for Why on earth you treated this case
// separately in the first place.
template<typename T, typename A>
struct transpose_expression<row_view<matrix<T, A> > >
    : public vector_expression<transpose_expression<row_view<matrix<T, A> > >
                               > {
 private:
  using row_view_type = row_view<matrix<T, A> >;
  using self_type = transpose_expression<row_view<matrix<T, A> > >;

 public:
    // public types.
  using value_type = typename row_view_type::value_type;
  using size_type = typename row_view_type::size_type;
  using difference_type = typename row_view_type::difference_type;
  using const_reference = typename row_view_type::const_reference;
  using reference = typename row_view_type::reference;
  using const_pointer = typename row_view_type::const_pointer;
  using pointer = typename row_view_type::pointer;
  using shape_type = typename row_view_type::shape_type;

  static constexpr bool is_vector = true;

  const row_view_type& rw;

  explicit transpose_expression(const row_view_type& rw) : rw(rw) {}

  inline size_type num_rows() const { return rw.num_cols(); }
  inline size_type num_cols() const { return rw.num_rows(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }
  inline size_type size() const { return rw.size(); }

  inline row_view<self_type> row_at(size_type row_index) {
    return row_view<self_type>(this, row_index);
  }

  inline col_view<self_type> col_at(size_type col_index) {
    return col_view<self_type>(this, col_index);
  }

  // Iterators.
  using iterator = typename row_view_type::const_iterator;
  using const_iterator = typename row_view_type::const_iterator;

  inline const_iterator begin() const { return rw.cbegin(); }
  inline const_iterator cbegin() const { return rw.cbegin(); }

  inline const_iterator end() const { return rw.cend(); }
  inline const_iterator cend() const { return rw.cend(); }
};

template<typename E>
inline
transpose_expression<E> transpose(const matrix_expression<E>& expr) {
  return transpose_expression<E>(expr.self());
}

template<typename E>
inline
transpose_expression<E> transpose(const vector_expression<E>& expr) {
  return transpose_expression<E>(expr.self());
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TRANSPOSE_EXPRESSION_H_
