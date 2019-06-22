// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_TRANSPOSE_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_TRANSPOSE_EXPRESSION_H_

#include <iterator>

namespace insight {

namespace linalg_detail {

// transpose iterator adaptor.
template<typename Iter>
class transpose_iterator {
 private:
  using iter_traits = std::iterator_traits<Iter>;

 public:
  using iterator_type = Iter;
  using iterator_category = typename iter_traits::iterator_category;
  using value_type = typename iter_traits::value_type;
  using difference_type = typename iter_traits::difference_type;
  using pointer = typename iter_traits::pointer;
  using reference = typename iter_traits::reference;

  transpose_iterator() : base_begin_(),
                         it_(),
                         index_(),
                         base_row_count_(),
                         base_col_count_() {}

  transpose_iterator(const Iter& it,
                     difference_type index,
                     difference_type base_row_count,
                     difference_type base_col_count)
      : base_begin_(it),
        it_(it),
        index_(index),
        base_row_count_(base_row_count),
        base_col_count_(base_col_count) {
  }

  template<typename U>
  transpose_iterator(const transpose_iterator<U>& src)
      : base_begin_(src.base_begin()),
        it_(src.base()),
        index_(src.index()),
        base_row_count_(src.base_row_count()),
        base_col_count_(src.base_col_count()) {
  }

  Iter base_begin() const { return base_begin_; }
  Iter base() const { return it_; }
  difference_type base_row_count() const { return base_row_count_; }
  difference_type base_col_count() const { return base_col_count_; }
  difference_type index() const { return index_; }

  reference operator*() const { return *it_; }
  pointer  operator->() const { return it_.operator->(); }

  transpose_iterator& operator++() {
    increment_();
    return *this;
  }

  transpose_iterator  operator++(int) {
    transpose_iterator tmp(*this);
    increment_();
    return tmp;
  }

  transpose_iterator& operator--() {
    decrement_();
    return *this;
  }

  transpose_iterator  operator--(int) {
    transpose_iterator tmp(*this);
    decrement_();
    return tmp;
  }

  transpose_iterator  operator+ (difference_type n) const {
    transpose_iterator tmp(*this);
    tmp += n;
    return tmp;
  }

  transpose_iterator& operator+=(difference_type n) {
    increment_(n);
    return *this;
  }

  transpose_iterator  operator- (difference_type n) const {
    transpose_iterator tmp(*this);
    tmp -= n;
    return tmp;
  }

  transpose_iterator& operator-=(difference_type n) {
    decrement_(n);
    return *this;
  }

  reference operator[](difference_type n) const {
    return static_cast<reference>((it_[to_base_index_(n)]));
  }

 private:
  const iterator_type base_begin_;
  iterator_type it_;
  difference_type index_;

  // TODO(Linh): This could be a potential problem in the future because
  // row_count and col_count of matrix are of type size_type (which is
  // typically size_t).
  difference_type base_row_count_;
  difference_type base_col_count_;

  void increment_();
  void increment_(difference_type n);
  void decrement_();
  void decrement_(difference_type n);
  difference_type to_base_index_(difference_type index) const;
};


template<typename Iter>
typename transpose_iterator<Iter>::difference_type
transpose_iterator<Iter>::to_base_index_(difference_type index) const {
  // Determine the current position in the transpose matrix given the
  // element's index.
  // Note: index = i * number_of_columns + j.
  //             = i * base_row_count_ + j
  difference_type i = index / base_row_count_;
  difference_type j = index % base_row_count_;
  difference_type base_size = base_row_count_ * base_col_count_;

  if (index >= base_size) {
    // TODO(Linh): This is an out of bounds index, should we return
    // base_size instead?
    return base_size + (index % base_size);
  } else if (index <= 0) {
    // TODO(Linh): Is index < 0, is is out of bounds. In that case, should
    // we just return 0?
    return index;
  } else {
    // Since the (i,j) element in the transposed matrix corresponds to the
    // element (j, i) in the original matrix, we have:
    return j * base_col_count_ + i;
  }
}

template<typename Iter>
void
transpose_iterator<Iter>::increment_() {
  ++index_;
  it_ = std::next(base_begin_, to_base_index_(index_));
}

template<typename Iter>
void
transpose_iterator<Iter>::increment_(difference_type n) {
  index_ += n;
  it_ = std::next(base_begin_, to_base_index_(index_));
}

template<typename Iter>
void
transpose_iterator<Iter>::decrement_() {
  --index_;
  it_ = std::next(base_begin_, to_base_index_(index_));
}

template<typename Iter>
void
transpose_iterator<Iter>::decrement_(difference_type n) {
  index_ -= n;
  it_ = std::next(base_begin_, to_base_index_(index_));
}

template<typename Iter1, typename Iter2>
inline
bool
operator==(const transpose_iterator<Iter1>& x,
           const transpose_iterator<Iter2>& y) {
  return (x.base_begin() == y.base_begin()) && (x.index() == y.index());
}

template<typename Iter1, typename Iter2>
inline
bool
operator!=(const transpose_iterator<Iter1>& x,
           const transpose_iterator<Iter2>& y) {
  return (x.base_begin() != y.base_begin()) || (x.index() != y.index());
}

template<typename Iter1, typename Iter2>
inline
bool
operator<(const transpose_iterator<Iter1>& x,
          const transpose_iterator<Iter2>& y) {
  return (x.base_begin() == y.base_begin()) && (x.index() < y.index());
}

template<typename Iter1, typename Iter2>
inline
bool
operator<=(const transpose_iterator<Iter1>& x,
           const transpose_iterator<Iter2>& y) {
  return (x.base_begin() == y.base_begin()) && (x.index() <= y.index());
}

template<typename Iter1, typename Iter2>
inline
bool
operator>(const transpose_iterator<Iter1>& x,
          const transpose_iterator<Iter2>& y) {
  return (x.base_begin() == y.base_begin()) && (x.index() > y.index());
}

template<typename Iter1, typename Iter2>
inline
bool
operator>=(const transpose_iterator<Iter1>& x,
           const transpose_iterator<Iter2>& y) {
  return (x.base_begin() == y.base_begin()) && (x.index() >= y.index());
}

template<typename Iter1, typename Iter2>
inline
auto
operator-(const transpose_iterator<Iter1>& x,
          const transpose_iterator<Iter2>& y)
    -> decltype((x.index() - y.index())) {
  return x.index() - y.index();
}

template<typename Iter>
inline
transpose_iterator<Iter>
operator+(typename transpose_iterator<Iter>::difference_type n,
          const transpose_iterator<Iter>& x) {
  transpose_iterator<Iter> tmp(x);
  tmp += n;
  return tmp;
}

template<typename Iter>
inline
transpose_iterator<Iter>
make_transpose_iterator(const Iter& it,
                        typename
                        transpose_iterator<Iter>::difference_type
                        index,
                        typename
                        transpose_iterator<Iter>::difference_type
                        base_row_count,
                        typename
                        transpose_iterator<Iter>::difference_type
                        base_col_count) {
  return transpose_iterator<Iter>(it, index, base_row_count, base_col_count);
}

// Forward declarations
template<typename Derived> struct matrix_expression;
template<typename Derived> struct vector_expression;
template<typename E> struct row_view;
template<typename E> struct col_view;

// Transpose of a matrix expression.
template<typename E>
class transpose_expression
    : public matrix_expression<transpose_expression<E> > {
 private:
  using self = transpose_expression<E>;
  using iter_traits = std::iterator_traits<typename E::iterator>;
  using iterator_ = typename E::iterator;
  using const_iterator_ = typename E::const_iterator;
 public:
  using value_type = typename iter_traits::value_type;
  using reference = typename iter_traits::reference;
  using size_type = typename E::size_type;

  using iterator = typename std::conditional<
    std::is_base_of<vector_expression<E>, E>::value,
    iterator_,
    transpose_iterator<iterator_>>::type;

  using const_iterator = typename std::conditional<
    std::is_base_of<vector_expression<E>, E>::value,
    const_iterator_,
    transpose_iterator<const_iterator_>>::type;

  using shape_type = typename E::shape_type;

  const E& e;

  explicit transpose_expression(const E& e) : e(e) {}

  inline size_type row_count() const { return e.col_count(); }
  inline size_type col_count() const { return e.row_count(); }
  inline size_type size() const { return e.size(); }
  inline shape_type shape() const {
    return shape_type(row_count(), col_count());
  }

  inline row_view<self> row_at(size_type row_index) {
    return row_view<self>(this, row_index);
  }

  inline col_view<self> col_at(size_type col_index) {
    return col_view<self>(this, col_index);
  }

  // TODO(Linh): Should we return e instead?
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  inline iterator begin() {
    return begin_(std::integral_constant<bool,
                  std::is_base_of<vector_expression<E>, E>::value>());
  }

  inline const_iterator begin() const {
    return cbegin_(std::integral_constant<bool,
                   std::is_base_of<vector_expression<E>, E>::value>());
  }

  inline const_iterator cbegin() const {
    return cbegin_(std::integral_constant<bool,
                   std::is_base_of<vector_expression<E>, E>::value>());
  }

  inline iterator end() {
    return end_(std::integral_constant<bool,
                std::is_base_of<vector_expression<E>, E>::value>());
  }

  inline const_iterator end() const {
    return cend_(std::integral_constant<bool,
                 std::is_base_of<vector_expression<E>, E>::value>());
  }

  inline const_iterator cend() const {
    return cend_(std::integral_constant<bool,
                 std::is_base_of<vector_expression<E>, E>::value>());
  }

 private:
  inline iterator begin_(std::true_type) {
    return e.begin();
  }

  inline iterator begin_(std::false_type) {
    return make_transpose_iterator(e.begin(), 0, e.row_count(),
                                   e.col_count());
  }

  inline const_iterator cbegin_(std::true_type) const {
    return e.cbegin();
  }

  inline const_iterator cbegin_(std::false_type) const {
    return make_transpose_iterator(e.cbegin(), 0, e.row_count(),
                                   e.col_count());
  }

  inline iterator end_(std::true_type) {
    return e.end();
  }

  inline iterator end_(std::false_type) {
    return make_transpose_iterator(e.begin(), e.size(), e.row_count(),
                                   e.col_count());
  }

  inline const_iterator cend_(std::true_type) const {
    return e.cend();
  }

  inline const_iterator cend_(std::false_type) const {
    return make_transpose_iterator(e.cbegin(), e.size(), e.row_count(),
                                   e.col_count());
  }
};

// Transpose of a row view. The reason why we need this specification are
// follows:
//
// 1. When transpose of row view (a row vector), we would expect to get
//    a column vector, that's why it is inherited from vector_expression
//    not matrix_expression.
//
// 2. This would allow us to view the transpose of a row view of a dense
//    matrix as a dense column vector, so that we could use BLAS for
//    some operations.
template<typename E>
class transpose_expression<row_view<E> >
    : public vector_expression<transpose_expression<row_view<E> > > {
 private:
  using self = transpose_expression<row_view<E> >;
  using iter_traits = std::iterator_traits<typename row_view<E>::iterator>;

 public:
  using value_type = typename iter_traits::value_type;
  using reference = typename iter_traits::reference;
  using size_type = typename row_view<E>::size_type;
  using iterator = typename row_view<E>::iterator;
  using const_iterator = typename row_view<E>::const_iterator;
  using shape_type = typename row_view<E>::shape_type;

  const row_view<E>& e;

  explicit transpose_expression(const row_view<E>& e) : e(e) { }

  inline size_type row_count() const { return e.col_count(); }
  inline size_type col_count() const { return e.row_count(); }
  inline size_type size() const { return e.size(); }
  inline shape_type shape() const {
    return shape_type(row_count(), col_count());
  }

  inline row_view<self> row_at(size_type row_index) {
    return row_view<self>(this, row_index);
  }

  inline col_view<self> col_at(size_type col_index) {
    return col_view<self>(this, col_index);
  }

  // TODO(Linh): Should we return e instead?
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  inline iterator begin() { return e.begin(); }
  inline const_iterator begin() const { return e.cbegin(); }
  inline const_iterator cbegin() const { return e.cbegin(); }

  inline iterator end() { return e.end(); }
  inline const_iterator end() const { return e.cend(); }
  inline const_iterator cend() const { return e.cend(); }
};
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_TRANSPOSE_EXPRESSION_H_
