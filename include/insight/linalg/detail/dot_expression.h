// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DOT_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_DOT_EXPRESSION_H_

#include <iterator>
#include <numeric>

namespace insight {

// Forward declarations
template<typename Derived> struct matrix_expression;
template<typename Derived> struct vector_expression;
template<typename E> struct row_view;
template<typename E> struct col_view;
template<typename E> struct transpose_expression;

template<typename M, typename V, typename Enable = void>
struct dot_expression;

// matrix-vector multiplication.
template<typename M, typename V>
struct dot_expression<
  M,
  V,
  typename std::enable_if<
    std::is_base_of<matrix_expression<M>, M>::value &&
    std::is_base_of<vector_expression<V>, V>::value &&
    std::is_same<typename M::value_type, typename V::value_type>::value,
                           void>::type
  >: public vector_expression<dot_expression<M, V> > {
 private:
  using self_type = dot_expression<M, V>;

 public:
  // public types.
  using value_type = typename M::value_type;
  using size_type = typename M::size_type;
  using difference_type = typename M::difference_type;
  using const_reference = value_type;  // typename M::const_reference;
  using reference = const_reference;
  using const_pointer = typename M::const_pointer;
  using pointer = const_pointer;
  using shape_type = typename M::shape_type;

  static constexpr bool is_vector = true;

  const M& m;
  const V& v;

  dot_expression(const M& m, const V& v) : m(m), v(v) {}

  inline size_type num_rows() const { return m.num_rows(); }
  inline size_type num_cols() const { return /*one*/v.num_cols(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }
  inline size_type size() const { return num_rows() * num_cols(); }

  inline row_view<self_type> row_at(size_type row_index) {
    return row_view<self_type>(this, row_index);
  }

  inline col_view<self_type> col_at(size_type col_index) {
    return col_view<self_type>(this, col_index);
  }

  // Transpose of this expression.
  inline transpose_expression<self_type> t() const {
    return transpose_expression<self_type>(*this);
  }

  // Iterator.

  class const_iterator;
  using iterator = const_iterator;

  class const_iterator {
   private:
    using const_matrix_iterator = typename M::const_iterator;
    using const_vector_iterator = typename V::const_iterator;

   public:
    // public types.
    using value_type = typename dot_expression::value_type;
    using difference_type = typename dot_expression::difference_type;
    using pointer = typename dot_expression::const_pointer;
    using reference = typename dot_expression::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : row_index_(),
                       vsize_(),
                       vbegin_(),
                       vend_(),
                       mit_() {}

    const_iterator(const dot_expression& expr, size_type row_index)
        : row_index_(row_index),
          vsize_(expr.v.size()),
          vbegin_(expr.v.begin()),
          vend_(expr.v.end()),
          mit_(expr.m.begin()) {}

    // Copy constructor.
    const_iterator(const const_iterator& it)
        : row_index_(it.row_index_),
          vsize_(it.vsize_),
          vbegin_(it.vbegin_),
          vend_(it.vend_),
          mit_(it.mit_) {
    }

    // Assignment operator.
    const_iterator& operator=(const const_iterator& it) {
      if (this == &it) { return *this; }
      row_index_ = it.row_index_;
      vsize_ = it.vsize_;
      vbegin_ = it.vbegin_;
      vend_ = it.vend_;
      mit_ = it.mit_;
      return *this;
    }

    // Dereference.
    inline reference operator*() const {
      return std::inner_product(vbegin_, vend_, mit_, /*zero*/value_type());
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) const {
      return (row_index_ == it.row_index_);
    }

    inline bool operator!=(const const_iterator& it) const {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline const_iterator& operator++() {
      ++row_index_;
      std::advance(mit_, vsize_);
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
    size_type vsize_;
    const_vector_iterator vbegin_;
    const_vector_iterator vend_;
    const_matrix_iterator mit_;
  };

  inline const_iterator begin() const { return const_iterator(*this, 0); }
  inline const_iterator cbegin() const { return begin(); }
  inline const_iterator end() const { return const_iterator(*this, size()); }
  inline const_iterator cend() const  {return end(); }
};


}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DOT_EXPRESSION_H_
