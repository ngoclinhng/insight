// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_MATRIX_MUL_VECTOR_H_
#define INCLUDE_INSIGHT_LINALG_MATRIX_MUL_VECTOR_H_

#include <iterator>
#include <numeric>

#include "insight/linalg/arithmetic_expression.h"

namespace insight {

template<
  typename M,
  typename V,
  typename std::enable_if<
    std::is_base_of<matrix_expression<M>, M>::value &&
    std::is_base_of<vector_expression<V>, V>::value &&
    std::is_same<typename M::value_type, typename V::value_type>::value,
    int>::type = 0
  >
struct matrix_mul_vector
    : public vector_expression<matrix_mul_vector<M, V> > {
  // public types.
  using value_type = typename M::value_type;
  using size_type = typename M::size_type;
  using difference_type = typename M::difference_type;
  using const_reference = typename M::const_reference;
  using reference = const_reference;
  using const_pointer = typename M::const_pointer;
  using pointer = const_pointer;
  using shape_type = typename M::shape_type;

  static constexpr bool is_vector = true;

  const M& m;
  const V& v;

  matrix_mul_vector(const M& m, const V& v) : m(m), v(v) {
    CHECK_EQ(m.num_cols(), v.num_rows());
  }

  inline size_type num_rows() const { return m.num_rows(); }
  inline size_type num_cols() const { return /*one*/v.num_cols(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }
  inline size_type size() const { return num_rows() * num_cols(); }

  // Iterator.

  class const_iterator;
  using iterator = const_iterator;

  class const_iterator {
   private:
    using const_matrix_iterator = typename M::const_iterator;
    using const_vector_iterator = typename V::const_iterator;

   public:
    // public types.
    using value_type = typename V::value_type;
    using difference_type = typename V::difference_type;
    using pointer = typename V::const_pointer;
    using reference = typename V::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : row_index_(),
                       vsize_(),
                       vbegin_(),
                       vend_(),
                       mit_() {}

    const_iterator(const matrix_mul_vector& expr, size_type row_index)
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
    inline value_type operator*() const {
      return std::inner_product(vbegin_, vend_, mit_, /*zero*/value_type());
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) {
      return (row_index_ == it.row_index_);
    }

    inline bool operator!=(const const_iterator& it) {
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

// matrix-vector multiplication.
template<typename M, typename V>
inline
matrix_mul_vector<M, V>
mat_mul(const matrix_expression<M>& me, const vector_expression<V>& ve) {
  return matrix_mul_vector<M, V>(me.self(), ve.self());
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_MATRIX_MUL_VECTOR_H_
