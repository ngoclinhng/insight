// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_MATMUL_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_MATMUL_EXPRESSION_H_

#include <numeric>
#include <iterator>
#include <type_traits>

#include "glog/logging.h"

namespace insight {
namespace linalg_detail {

template<typename Derived> struct vector_expression;
template<typename Derived> struct matrix_expression;

template<typename MatrixIter, typename VectorIter>
class matrix_vector_multiplication_iterator;

template<typename ME, typename VE, typename Enable = void>
class matmul_expression;

// multiplication between a generic matrix expression and a generic vector
// expression.
template<typename ME, typename VE>
class matmul_expression<
  ME,
  VE,
  typename std::enable_if<
    std::is_base_of<matrix_expression<ME>, ME>::value &&
    std::is_base_of<vector_expression<VE>, VE>::value &&
    std::is_same<typename ME::value_type, typename VE::value_type>::value,
    void>::type>
    : public vector_expression<matmul_expression<ME, VE> > {
 public:
  using value_type = typename ME::value_type;
  using reference = value_type;
  using size_type = typename ME::size_type;
  using shape_type = typename ME::shape_type;
  using const_iterator =
      matrix_vector_multiplication_iterator<typename ME::const_iterator,
                                            typename VE::const_iterator>;
  using iterator = const_iterator;

  const ME& m;
  const VE& v;

  matmul_expression(const ME& m, const VE& v) : m(m), v(v) {
    CHECK_EQ(m.col_count(), v.size()) << "matmul: mismatched dimensions";
  }

  inline size_type row_count() const { return m.row_count(); }
  inline size_type col_count() const { return v.col_count()/*one*/; }
  inline size_type size() const { return row_count() * col_count(); }
  inline shape_type shape() const {
    return shape_type(row_count(), col_count());
  }

  inline const_iterator begin() const {
    return const_iterator(m.cbegin(), 0, v.cbegin(), v.cend());
  }

  inline const_iterator cbegin() const {
    return const_iterator(m.cbegin(), 0, v.cbegin(), v.cend());
  }

  inline const_iterator end() const {
    return const_iterator(m.cend(), m.row_count(), v.cbegin(), v.cend());
  }

  inline const_iterator cend() const {
    return const_iterator(m.cend(), m.row_count(), v.cbegin(), v.cend());
  }
};

template<typename MatrixIter, typename VectorIter>
class matrix_vector_multiplication_iterator {
 private:
  using matrix_iter_traits = std::iterator_traits<MatrixIter>;
  using vector_iter_traits = std::iterator_traits<VectorIter>;

 public:
  using matrix_iterator_type = MatrixIter;
  using vector_iterator_type = VectorIter;

  static_assert(std::is_same<typename matrix_iter_traits::iterator_category,
                typename vector_iter_traits::iterator_category>::value,
                "MatrixIter and VectorIter must have the same "
                "iterator_category");

  static_assert(std::is_same<typename matrix_iter_traits::difference_type,
                typename vector_iter_traits::difference_type>::value,
                "MatrixIter and VectorIter must have the same "
                "difference_type");

  static_assert(std::is_same<typename matrix_iter_traits::value_type,
                typename vector_iter_traits::value_type>::value,
                "MatrixIter and VectorIter must have the same "
                "value_type");

  using iterator_category = typename matrix_iter_traits::iterator_category;
  using value_type = typename matrix_iter_traits::value_type;
  using difference_type = typename matrix_iter_traits::difference_type;
  using pointer = void;
  using reference = value_type;

  matrix_vector_multiplication_iterator()
      : row_start_(),
        row_index_(),
        vec_begin_(),
        vec_end_(),
        vec_size_() {
  }

  matrix_vector_multiplication_iterator(matrix_iterator_type row_start,
                                        difference_type row_index,
                                        vector_iterator_type vec_begin,
                                        vector_iterator_type vec_end)
      : row_start_(row_start),
        row_index_(row_index),
        vec_begin_(vec_begin),
        vec_end_(vec_end),
        vec_size_(std::distance(vec_begin, vec_end)) { }

  template<typename MI, typename VI>
  matrix_vector_multiplication_iterator(
      const matrix_vector_multiplication_iterator<MI, VI>& it)
      : row_start_(it.row_start()),
        row_index_(it.row_index()),
        vec_begin_(it.vec_begin()),
        vec_end_(it.vec_end()),
        vec_size_(it.vec_size()) {}

  matrix_iterator_type row_start() const { return row_start_; }
  difference_type row_index() const { return row_index_; }
  vector_iterator_type vec_begin() const { return vec_begin_; }
  vector_iterator_type vec_end() const { return vec_end_; }
  difference_type vec_size() const {return vec_size_; }


  reference operator*() const {
    return std::inner_product(vec_begin_, vec_end_, row_start_,
                              /*zero*/value_type());
  }

  // TODO(Linh): neccessary?
  // pointer  operator->() const { return it_;}

  matrix_vector_multiplication_iterator& operator++() {
    std::advance(row_start_, vec_size_);
    ++row_index_;
    return *this;
  }

  matrix_vector_multiplication_iterator  operator++(int) {
    matrix_vector_multiplication_iterator tmp(*this);
    std::advance(row_start_, vec_size_);
    ++row_index_;
    return tmp;
  }

  matrix_vector_multiplication_iterator& operator--() {
    std::advance(row_start_, -vec_size_);
    --row_index_;
    return *this;
  }

  matrix_vector_multiplication_iterator  operator--(int) {
    matrix_vector_multiplication_iterator tmp(*this);
    std::advance(row_start_, -vec_size_);
    --row_index_;
    return tmp;
  }

  matrix_vector_multiplication_iterator  operator+ (difference_type n) const {
    return matrix_vector_multiplication_iterator(
        std::next(row_start_, n * vec_size_), row_index_ + n,
        vec_begin_, vec_end_);
  }

  matrix_vector_multiplication_iterator& operator+=(difference_type n) {
    std::advance(row_start_, n * vec_size_);
    row_index_ += n;
    return *this;
  }

  matrix_vector_multiplication_iterator  operator- (difference_type n) const {
    return matrix_vector_multiplication_iterator(
        std::prev(row_start_, n * vec_size_), row_index_ - n,
        vec_begin_, vec_end_);
  }

  matrix_vector_multiplication_iterator& operator-=(difference_type n) {
    std::advance(row_start_, -n * vec_size_);
    row_index_ -= n;
    return *this;
  }

  reference operator[](difference_type n) const {
    return std::inner_product(vec_begin_, vec_end_,
                              std::next(row_start_, n * vec_size_),
                              /*zero*/value_type());
  }

 private:
  matrix_iterator_type row_start_;
  difference_type row_index_;
  vector_iterator_type vec_begin_;
  vector_iterator_type vec_end_;
  difference_type vec_size_;
};

template<typename T1, typename T2, typename U1, typename U2>
inline
bool
operator==(const matrix_vector_multiplication_iterator<T1, T2>& x,
           const matrix_vector_multiplication_iterator<U1, U2>& y) {
  return (x.vec_begin() == y.vec_begin()) && (x.vec_end() == y.vec_end()) &&
      (x.row_start() == y.row_start());
}

template<typename T1, typename T2, typename U1, typename U2>
inline
bool
operator!=(const matrix_vector_multiplication_iterator<T1, T2>& x,
           const matrix_vector_multiplication_iterator<U1, U2>& y) {
  return (x.vec_begin() != y.vec_begin()) || (x.vec_end() != y.vec_end()) ||
      (x.row_start() != y.row_start());
}


template<typename T1, typename T2, typename U1, typename U2>
inline
bool
operator<(const matrix_vector_multiplication_iterator<T1, T2>& x,
           const matrix_vector_multiplication_iterator<U1, U2>& y) {
  return (x.vec_begin() == y.vec_begin()) && (x.vec_end() == y.vec_end()) &&
      (x.row_start() < y.row_start());
}

template<typename T1, typename T2, typename U1, typename U2>
inline
bool
operator<=(const matrix_vector_multiplication_iterator<T1, T2>& x,
           const matrix_vector_multiplication_iterator<U1, U2>& y) {
  return (x.vec_begin() == y.vec_begin()) && (x.vec_end() == y.vec_end()) &&
      (x.row_start() <= y.row_start());
}

template<typename T1, typename T2, typename U1, typename U2>
inline
bool
operator>(const matrix_vector_multiplication_iterator<T1, T2>& x,
          const matrix_vector_multiplication_iterator<U1, U2>& y) {
  return (x.vec_begin() == y.vec_begin()) && (x.vec_end() == y.vec_end()) &&
      (x.row_start() > y.row_start());
}

template<typename T1, typename T2, typename U1, typename U2>
inline
bool
operator>=(const matrix_vector_multiplication_iterator<T1, T2>& x,
           const matrix_vector_multiplication_iterator<U1, U2>& y) {
  return (x.vec_begin() == y.vec_begin()) && (x.vec_end() == y.vec_end()) &&
      (x.row_start() >= y.row_start());
}

template<typename T1, typename T2, typename U1, typename U2>
inline
auto
operator-(const matrix_vector_multiplication_iterator<T1, T2>& x,
          const matrix_vector_multiplication_iterator<U1, U2>& y)
    -> decltype(x.row_index() - y.row_index()) {
  return x.row_index() - y.row_index();
}

template<typename MatrixIter, typename VectorIter>
inline
matrix_vector_multiplication_iterator<MatrixIter, VectorIter>
operator+(typename matrix_vector_multiplication_iterator<MatrixIter,
          VectorIter>::difference_type n,
          const matrix_vector_multiplication_iterator<MatrixIter,
          VectorIter>& x) {
  return matrix_vector_multiplication_iterator<MatrixIter, VectorIter>(
      std::next(x.row_start(), n * x.vec_size()),
      x.row_index() + n,
      x.vec_begin(), x.vec_end());
}
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_MATMUL_EXPRESSION_H_
