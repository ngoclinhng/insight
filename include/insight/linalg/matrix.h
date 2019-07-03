// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_MATRIX_H_
#define INCLUDE_INSIGHT_LINALG_MATRIX_H_

#include <algorithm>
#include <limits>
#include <utility>
#include <initializer_list>

#include "insight/memory.h"

#include "insight/linalg/detail/row_view.h"
#include "insight/linalg/detail/col_view.h"
#include "insight/linalg/detail/expression_evaluator.h"
#include "insight/linalg/detail//dense_base.h"
#include "insight/linalg/detail/blas_routines.h"

#include "glog/logging.h"

namespace insight {

// Dense, row-major order matrix.
template<typename T, typename Alloc = allocator<T> >  // NOLINT
class matrix
    : private linalg_detail::dense_base<T, Alloc>,
  public linalg_detail::matrix_expression<matrix<T, Alloc> > {
 private:
  using base = linalg_detail::dense_base<T, Alloc>;
  using self = matrix;
  using alloc_traits = typename base::alloc_traits;

 public:
  using value_type = T;
  using allocator_type = Alloc;
  using reference = typename base::reference;
  using const_reference = typename base::const_reference;
  using size_type = typename base::size_type;
  using shape_type = std::pair<size_type, size_type>;  // NOLINT
  using difference_type = typename base::difference_type;
  using pointer = typename base::pointer;
  using const_pointer = typename base::const_pointer;
  using iterator = pointer;
  using const_iterator = const_pointer;

  static_assert(std::is_same<typename allocator_type::value_type,
                value_type>::value,
                "Alloc::value_type must be the same as value_type");

  // TODO(Linh): What about complex data type?
  static_assert(std::is_arithmetic<value_type>::value,
                "matrix<T> only accepts arithmetic types.");

  // Constructs an empty matrix. An empty matrix is a matrix with zero number
  // of rows, zero number of columns, and zero number of elements.
  matrix() INSIGHT_NOEXCEPT_IF(
      std::is_nothrow_default_constructible<allocator_type>::value);

  // Constructs a dense matrix with row_count (or dim.first) number of rows,
  // and col_count (or dim.second) number of columns, all elements are
  // default-constructed.
  //
  // If row_count * col_count == 0 or dim.first * dim.second == 0,
  // then an empty matrix will be constructed.
  matrix(size_type row_count, size_type col_count);
  explicit matrix(const shape_type& dim);

  // Constructs a dense matrix with row_count (or dim.first) number of rows,
  // and col_count (or dim.second) number of columns, all elements are
  // copy-constructed from value.
  //
  // If row_count * col_count == 0 or dim.first * dim.second == 0,
  // then an empty matrix will be constructed.
  matrix(size_type row_count, size_type col_count, const_reference value);
  matrix(const shape_type& dim, const_reference value);

  // Constructs a 1 by n matrix (a row vector) with the contents in the
  // range [first, last) where n is the number of elements in that range.
  //
  // If std::distance(first, last) == 0, then an empty matrix will be
  // constructed.
  template<typename ForwardIter>
  matrix(ForwardIter first, ForwardIter last,
         typename std::enable_if<
         internal::is_forward_iterator<ForwardIter>::value &&
         std::is_constructible<value_type,
         typename std::iterator_traits<ForwardIter>::reference>::value
         >::type* = 0);

  // TODO(Linh): Should this be default, i.e = default instead?
  ~matrix() {}

  // Copy constructor and assignment operator.
  matrix(const matrix& m);
  matrix& operator=(const matrix& m);

  // Move constructor & move assignment operator.
  matrix(matrix&& m) INSIGHT_NOEXCEPT_IF(
      std::is_nothrow_move_constructible<allocator_type>::value);
  matrix& operator=(matrix&& m) INSIGHT_NOEXCEPT_IF(
      alloc_traits::propagate_on_container_move_assignment::value &&
      std::is_nothrow_move_assignable<allocator_type>::value);

  // Constructs a 1 by n matrix (a row vector) with the contents of the
  // initializer_list il where n == il.size().
  //
  // If n == 0, then an empty matrix will be contructed.
  matrix(std::initializer_list<value_type> il);
  matrix& operator=(std::initializer_list<value_type> il);

  // Constructs a `m` by `n` matrix with the contents of the initializer
  // list `init`, where `m` is the size of `il` and `n` is the size of
  // each of its sub-list.
  matrix(std::initializer_list<std::initializer_list<value_type> > il);
  matrix& operator=(std::initializer_list<std::initializer_list<value_type> > il);  // NOLINT

  // Constructs a matrix from a generic matrix expression.

  template<typename E>
  matrix(const linalg_detail::matrix_expression<E>& expr);  // NOLINT

  template<typename E>
  matrix& operator=(const linalg_detail::matrix_expression<E>& expr);

  // Returns the allocator.
  allocator_type get_allocator() const INSIGHT_NOEXCEPT {
    return this->alloc_;
  }

  // Iterators.

  inline iterator begin() INSIGHT_NOEXCEPT { return this->begin_; }
  inline const_iterator begin() const INSIGHT_NOEXCEPT {
    return this->begin_;
  }
  inline iterator end() INSIGHT_NOEXCEPT { return this->end_; }
  inline const_iterator end() const INSIGHT_NOEXCEPT { return this->end_; }

  inline const_iterator cbegin() const INSIGHT_NOEXCEPT { return begin(); }
  inline const_iterator cend() const INSIGHT_NOEXCEPT { return end(); }

  // Returns the number of rows in the matrix.
  inline size_type row_count() const INSIGHT_NOEXCEPT { return dim_.first; }

  // Returns the number of columns in the matrix.
  inline size_type col_count() const INSIGHT_NOEXCEPT { return dim_.second; }

  // Returns the shape of this matrix.
  inline const shape_type& shape() const INSIGHT_NOEXCEPT { return dim_; }

  // Returns the number of elements in the matrix.
  inline size_type size() const INSIGHT_NOEXCEPT { return base::size(); }

  // Returns the number of elements that the container has allocated space
  // for
  inline size_type capacity() const INSIGHT_NOEXCEPT {
    return base::capacity();
  }

  // Returns true if the matrix is empty, i.e there is no elements in the
  // matrix, and row_count() == col_count() == 0.
  inline bool empty() const INSIGHT_NOEXCEPT {
    return (this->begin_ == this->end_);
  }

  // Returns the maximum number of elements the matrix is able to hold.
  size_type max_size() const INSIGHT_NOEXCEPT;

  // Returns a reference to the element at the specified index.
  // No bounds checking is performed.
  inline reference operator[](size_type index) INSIGHT_NOEXCEPT {
    return this->begin_[index];
  }

  // Returns the const reference to the element at the specified index.
  // No bounds checking is performed.
  inline const_reference operator[](size_type index) const INSIGHT_NOEXCEPT {
    return this->begin_[index];
  }

  // Returns the reference to the element at the specified location
  // (row_index, col_index). No bounds checking is performed
  inline reference operator()(size_type row_index, size_type col_index)
      INSIGHT_NOEXCEPT {
    return this->begin_[row_index * dim_.second + col_index];
  }

  // Returns the const reference to the element at the specified location
  // (row_index, col_index). No bounds checking is performed
  inline const_reference operator()(size_type row_index, size_type col_index)
      const INSIGHT_NOEXCEPT {
    return this->begin_[row_index * dim_.second + col_index];
  }

  // Returns the pointer to the underlying array.
  inline value_type* data() INSIGHT_NOEXCEPT { return this->begin_; }
  inline const value_type* data() const INSIGHT_NOEXCEPT {
    return this->begin_;
  }

  // Clear all the contents in the matrix and set its size to zero.
  // Matrix will become an emtpy matrix after clear call.
  void clear() INSIGHT_NOEXCEPT;

  // Changes the dimension of the matrix without affecting the contents.
  void reshape(size_type new_row_count, size_type new_col_count)
      INSIGHT_NOEXCEPT;

  void swap(matrix& m) INSIGHT_NOEXCEPT_IF(
      !alloc_traits::propagate_on_container_swap::value ||
      internal::is_nothrow_swappable<allocator_type>::value);

  // Accesses the row at index `row_index`.
  inline linalg_detail::row_view<self> row_at(size_type row_index) {
    return linalg_detail::row_view<self>(this, row_index);
  }

  // Accesses the column at index `col_index`.
  inline linalg_detail::col_view<self> col_at(size_type col_index) {
    return linalg_detail::col_view<self>(this, col_index);
  }

  // return the transpose of this matrix.
  inline linalg_detail::transpose_expression<self> t() const {
    return linalg_detail::transpose_expression<self>(*this);
  }

  // Matrix-scalar arithmetic.

  // Increments each and every element in the matrix by the constant scalar.
  inline matrix& operator+=(const_reference scalar) {
    std::for_each(this->begin_, this->end_, [&](reference e) { e += scalar; });
    return *this;
  }

  // Decrements each and every element in the matrix by the constant scalar.
  inline matrix& operator-=(const_reference scalar) {
    std::for_each(this->begin_, this->end_, [&](reference e) { e -= scalar; });
    return *this;
  }

  // Replaces each and every element in the matrix by the result of
  // multiplication of that element and a scalar.
  inline matrix& operator*=(const_reference scalar) {
    mul_scalar_(scalar, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in the matrix by the result of
  // dividing that element by a constant scalar.
  inline matrix& operator/=(const_reference scalar) {
    div_scalar_(scalar, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // matrix-matrix arithmetic.

  // Replaces each and every element in this matrix by the result of
  // adding that element with the corresponfing element in the matrix m.
  inline matrix& operator+=(const matrix& m) {
    CHECK_EQ(row_count(), m.row_count());
    CHECK_EQ(col_count(), m.col_count());
    add_matrix_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in this matrix by the result of
  // substracting the corresponding element in the matrix m from that
  // element.
  inline matrix& operator-=(const matrix& m) {
    CHECK_EQ(row_count(), m.row_count());
    CHECK_EQ(col_count(), m.col_count());
    sub_matrix_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in this matrix by the result of
  // multiplying that element and the corresponding element in the matrix m.
  inline matrix& operator*=(const matrix& m) {
    CHECK_EQ(row_count(), m.row_count());
    CHECK_EQ(col_count(), m.col_count());
    mul_matrix_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in this matrix by the result of
  // dividing that element by the corresponding element in the matrix m.
  inline matrix& operator/=(const matrix& m) {
    CHECK_EQ(row_count(), m.row_count());
    CHECK_EQ(col_count(), m.col_count());
    div_matrix_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // matrix expression arithmetic.

  template<typename E>
  inline matrix& operator+=(const linalg_detail::matrix_expression<E>& expr) {
    CHECK_EQ(row_count(), expr.self().row_count());
    CHECK_EQ(col_count(), expr.self().col_count());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.add(this->begin_);
    return *this;
  }

  template<typename E>
  inline matrix& operator-=(const linalg_detail::matrix_expression<E>& expr) {
    CHECK_EQ(row_count(), expr.self().row_count());
    CHECK_EQ(col_count(), expr.self().col_count());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.sub(this->begin_);
    return *this;
  }

  template<typename E>
  inline matrix& operator*=(const linalg_detail::matrix_expression<E>& expr) {
    CHECK_EQ(row_count(), expr.self().row_count());
    CHECK_EQ(col_count(), expr.self().col_count());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.mul(this->begin_);
    return *this;
  }

  template<typename E>
  inline matrix& operator/=(const linalg_detail::matrix_expression<E>& expr) {
    CHECK_EQ(row_count(), expr.self().row_count());
    CHECK_EQ(col_count(), expr.self().col_count());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.div(this->begin_);
    return *this;
  }

 private:
  shape_type dim_;

  // Allocate space for n objects.
  void allocate_memory_(size_type n);

  // Deallocate memory.
  void deallocate_memory_() INSIGHT_NOEXCEPT;

  // Default constructs n objects starting at end_.
  void construct_at_end_(size_type n);

  // Copy constructs n objects starting at end_ from value.
  void construct_at_end_(size_type n, const_reference value);

  // Constructs n objects starting at end_ from the range [first, last).
  // where n == std::distance(first, last).
  template<typename ForwardIter>
  typename
  std::enable_if<internal::is_forward_iterator<ForwardIter>::value,
                 void>::type
  construct_at_end_(ForwardIter first, ForwardIter last);

  // Replaces all the contents in the buffer by the contents of the range
  // [first, last). Memory allocated if neccessary.
  template<typename ForwardIter>
  typename
  std::enable_if<
    internal::is_forward_iterator<ForwardIter>::value &&
    std::is_constructible<
      value_type,
      typename std::iterator_traits<ForwardIter>::reference>::value,
    void>::type
  assign_(ForwardIter first, ForwardIter last);

  // Move m to this matrix when propagate_on_container_move_assignment
  // is true. this's allocator will be replaced
  void move_assign_(matrix& m, std::true_type) INSIGHT_NOEXCEPT_IF(  // NOLINT
      std::is_nothrow_move_assignable<allocator_type>::value);

  // Move m to this matrix whenpropagate_on_container_move_assignment
  // is false. the old allocator is kept.
  void move_assign_(matrix& m, std::false_type);  // NOLINT

  // helper methods for matrix-scalar arithmetic.

  void mul_scalar_(const_reference scalar, std::true_type);
  void mul_scalar_(const_reference scalar, std::false_type);

  void div_scalar_(const_reference scalar, std::true_type);
  void div_scalar_(const_reference scalar, std::false_type);

  // Helper methods for matrix-matrix arithmetic.

  void add_matrix_(const matrix& m, std::true_type);
  void add_matrix_(const matrix& m, std::false_type);

  void sub_matrix_(const matrix& m, std::true_type);
  void sub_matrix_(const matrix& m, std::false_type);

  void mul_matrix_(const matrix& m, std::true_type);
  void mul_matrix_(const matrix& m, std::false_type);

  void div_matrix_(const matrix& m, std::true_type);
  void div_matrix_(const matrix& m, std::false_type);
};

template<typename T, typename Alloc>
inline
matrix<T, Alloc>::matrix() INSIGHT_NOEXCEPT_IF(
    std::is_nothrow_default_constructible<allocator_type>::value)
    : base(), dim_() {
}

template<typename T, typename Alloc>
matrix<T, Alloc>::matrix(size_type row_count, size_type col_count)
    : base(),
      dim_() {
  size_type sz = row_count * col_count;
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(sz);
    dim_ = std::make_pair(row_count, col_count);
  }
}

template<typename T, typename Alloc>
matrix<T, Alloc>::matrix(const shape_type& dim)
    : base(),
      dim_() {
  size_type sz = dim.first * dim.second;
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(sz);
    dim_ = dim;
  }
}

template<typename T, typename Alloc>
matrix<T, Alloc>::matrix(size_type row_count,
                         size_type col_count,
                         const_reference value)
    : base(),
      dim_() {
  size_type sz = row_count * col_count;
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(sz, value);
    dim_ = std::make_pair(row_count, col_count);
  }
}

template<typename T, typename Alloc>
matrix<T, Alloc>::matrix(const shape_type& dim,
                         const_reference value)
    : base(),
      dim_() {
  size_type sz = dim.first * dim.second;
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(sz, value);
    dim_ = dim;
  }
}

template<typename T, typename Alloc>
template<typename ForwardIter>
matrix<T, Alloc>::matrix(ForwardIter first, ForwardIter last,
                         typename std::enable_if<
                         internal::is_forward_iterator<ForwardIter>::value &&
                         std::is_constructible<value_type,
                         typename std::iterator_traits<ForwardIter>::reference>::value>::type*)
    : base(),
      dim_() {
  size_type sz = static_cast<size_type>(std::distance(first, last));
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(first, last);
    dim_ = std::make_pair(1, sz);
  }
}

template<typename T, typename Alloc>
matrix<T, Alloc>::matrix(const matrix& m)
    : base(alloc_traits::select_on_container_copy_construction(m.alloc_)),
      dim_(m.shape()) {
  size_type sz = m.size();
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(m.begin_, m.end_);
  }
}

template<typename T, typename Alloc>
inline
matrix<T, Alloc>&
matrix<T, Alloc>::operator=(const matrix& m) {
  if (this != &m) {
    base::copy_assign_alloc_(m);
    assign_(m.begin_, m.end_);
    dim_ = m.dim_;
  }
  return *this;
}

template<typename T, typename Alloc>
inline
matrix<T, Alloc>::matrix(matrix&& m) INSIGHT_NOEXCEPT_IF(
    std::is_nothrow_move_constructible<allocator_type>::value)
    :   base(std::move(m.alloc_)),
        dim_(std::move(m.dim_/*this cannot throw?*/)) {
  this->begin_ = m.begin_;
  this->end_ = m.end_;
  this->end_cap_ = m.end_cap_;
  m.begin_ = m.end_ = m.end_cap_ = nullptr;
}

// TODO(Linh): If propagate_on_container_move_assignment is false,
// then the `false` version of move_assign_ will be called, but
// for this version to be nothrow we need alloc_traits::is_always_equal is
// true, unforturenately this feature is only available since C++17?
template<typename T, typename Alloc>
inline
matrix<T, Alloc>&
matrix<T, Alloc>::operator=(matrix&& m) INSIGHT_NOEXCEPT_IF(
    alloc_traits::propagate_on_container_move_assignment::value &&
    std::is_nothrow_move_assignable<allocator_type>::value) {
  move_assign_(m, std::integral_constant<bool, alloc_traits::propagate_on_container_move_assignment::value>());  // NOLINT
  return *this;
}

template<typename T, typename Alloc>
inline
matrix<T, Alloc>::matrix(std::initializer_list<value_type> il)
    : base(),
      dim_() {
  size_type sz = il.size();
  if (sz > 0) {
    allocate_memory_(sz);
    construct_at_end_(il.begin(), il.end());
    dim_ = std::make_pair(1, sz);
  }
}

template<typename T, typename Alloc>
inline
matrix<T, Alloc>&
matrix<T, Alloc>::operator=(std::initializer_list<value_type> il) {
  assign_(il.begin(), il.end());
  dim_ = std::make_pair(il.size() > 0 ? 1 : 0, il.size());
  return *this;
}

template<typename T, typename Alloc>
matrix<T, Alloc>::matrix(std::initializer_list<std::initializer_list<value_type> > il)  // NOLINT
    : base(),
      dim_() {
  // Make sure all sub-lists are of the same size.
  size_type sublist_size = static_cast<size_type>(il.begin()->size());
  CHECK_GT(sublist_size, static_cast<size_type>(0));
  CHECK(std::all_of(il.begin(), il.end(),
                    [&](const std::initializer_list<value_type>& sublist) {
                      return (sublist.size() == sublist_size);
                    }))
      << "matrix: invalid nested initializer list: all sublists must have "
      << "the same number of elements";

  size_type sz = sublist_size * il.size();
  allocate_memory_(sz);
  for (auto it = il.begin(); it != il.end(); ++it) {
    construct_at_end_(it->begin(), it->end());
  }
  dim_ = std::make_pair(il.size(), sublist_size);
}

template<typename T, typename Alloc>
matrix<T, Alloc>&
matrix<T, Alloc>::operator=(std::initializer_list<std::initializer_list<value_type> > il) {  // NOLINT
  // Make sure all sub-lists are of the same size.
  size_type sublist_size = static_cast<size_type>(il.begin()->size());
  CHECK_GT(sublist_size, static_cast<size_type>(0));
  CHECK(std::all_of(il.begin(), il.end(),
                    [&](const std::initializer_list<value_type>& sublist) {
                      return (sublist.size() == sublist_size);
                    }))
      << "matrix: invalid nested initializer list: all sublists must have "
      << "the same number of elements";

  size_type new_size = il.size() * sublist_size;

  if (new_size <= capacity()) {
    // TODO(Linh): Technically speaking we have to contruct (not copy) from
    // the range [end_, end_cap_). But we're only dealing with arithmetic
    // types, right?
    pointer mid = this->begin_;
    for (auto it = il.begin(); it != il.end(); ++it) {
      mid = std::copy(it->begin(), it->end(), mid);
    }
    this->end_ = mid;
  } else {
    deallocate_memory_();
    allocate_memory_(new_size);
    for (auto it = il.begin(); it != il.end(); ++it) {
      construct_at_end_(it->begin(), it->end());
    }
  }
  dim_ = std::make_pair(il.size(), sublist_size);
  return *this;
}

template<typename T, typename Alloc>
template<typename E>
matrix<T, Alloc>::matrix(const linalg_detail::matrix_expression<E>& expr)
    : base(),
      dim_(expr.self().shape()) {
  size_type sz = expr.self().size();
  if (sz > 0) {
    allocate_memory_(sz);
    // construct_at_end_(expr.self().begin(), expr.self().end());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.assign(this->begin_);
    this->end_ = this->begin_ + sz;
  }
}

template<typename T, typename Alloc>
template<typename E>
matrix<T, Alloc>&
matrix<T, Alloc>::operator=(const linalg_detail::matrix_expression<E>& expr) {
  size_type new_size = expr.self().size();
  if (new_size > capacity()) {
    deallocate_memory_();
    allocate_memory_(new_size);
  }
  linalg_detail::expression_evaluator<E> evaluator(expr.self());
  evaluator.assign(this->begin_);
  this->end_ = this->begin_ + new_size;
  dim_ = expr.self().shape();
  return *this;
}

template<typename T, typename Alloc>
inline
typename matrix<T, Alloc>::size_type
matrix<T, Alloc>::max_size() const INSIGHT_NOEXCEPT {
  return std::min<size_type>(alloc_traits::max_size(this->alloc_),
                             std::numeric_limits<difference_type>::max());
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::clear() INSIGHT_NOEXCEPT {
  base::clear();
  dim_ = std::make_pair(0, 0);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::reshape(size_type new_row_count,
                          size_type new_col_count) INSIGHT_NOEXCEPT {
  if (!empty() && (new_row_count * new_col_count == size())) {
    dim_ = std::make_pair(new_row_count, new_col_count);
  }
}

template<typename T, typename Alloc>
void
matrix<T, Alloc>::swap(matrix& m) INSIGHT_NOEXCEPT_IF(
    !alloc_traits::propagate_on_container_swap::value ||
    internal::is_nothrow_swappable<allocator_type>::value) {
  DCHECK(alloc_traits::propagate_on_container_swap::value ||
        (this->alloc_ == m.alloc_))
      << "matrix::swap Either propagate_on_container_swap must be true "
      << "or the allocators must compare equal";
  using std::swap;
  swap(this->begin_, m.begin_);
  swap(this->end_, m.end_);
  swap(this->end_cap_, m.end_cap_);
  swap(dim_, m.dim_);  // TODO(Linh): nothrow?
  // The two allocators will be swapped iff propagate_on_container_swap is
  // true, otherwise nothing happens (that's why we need them compare
  // equal).
  swap_allocator(this->alloc_, m.alloc_, std::integral_constant<bool, alloc_traits::propagate_on_container_swap::value>());  // NOLINT
}

// Allocate space for n objects.
// throws length_error if n > max_size()
// throws (probablY bad_alloc) if memory run out
// Precondition: begin_ == end_ == end_cap_ == 0
// Precondition: n > 0
// Postcondition: capacity() == n
// Postcondition: size() == 0
template<typename T, typename Alloc>
void
matrix<T, Alloc>::allocate_memory_(size_type n) {
  if (n > max_size())
    this->throw_length_error("matrix::allocate_memory_(n): "
                             "the requested size n is too large");
  this->begin_ = this->end_ = alloc_traits::allocate(this->alloc_, n);
  this->end_cap_ = this->begin_ + n;
}

// Deallocate memory.
template<typename T, typename Alloc>
void
matrix<T, Alloc>::deallocate_memory_() INSIGHT_NOEXCEPT {
  if (this->begin_ != nullptr) {
    // TODO(Linh): Can we just skip the clear step since we're only dealing
    // with arithmetic types?
    clear();
    alloc_traits::deallocate(this->alloc_, this->begin_, capacity());
    this->begin_ = this->end_ = this->end_cap_ = nullptr;
    dim_ = std::make_pair(0, 0);  // TODO(Linh): neccessary?
  }
}

// Default constructs n objects starting at end_.
// throws if constructions throws.
// Precondition: n > 0
// Precondition: size() + n <= capacity()
// Postcondition: size() == size() + n
template<typename T, typename Alloc>
void
matrix<T, Alloc>::construct_at_end_(size_type n) {
  // TODO(Linh): Can we replace the do-while loop with std::fill since we're
  // only dealing with arithmetic types? How about std::uninitialized_fill?
  // do {
  //   alloc_traits::construct(this->alloc_, this->end_);
  //   ++this->end_;
  //   --n;
  // } while (n > 0);
  this->end_ = this->begin_ + n;
  std::fill(this->begin_, this->end_, value_type());
}

// Copy constructs n objects starting at end_ from value.
// throws if construction throws.
// Precondition: n > 0
// Precondition: size() + n <= capacity().
// Postcondition: size() = old size() + n
// Postcondition: [i] == value for all i in [size() - n, size()]
template<typename T, typename Alloc>
void
matrix<T, Alloc>::construct_at_end_(size_type n, const_reference value) {
  // TODO(Linh): Can we replace the do-while loop with std::fill since we're
  // only dealing with arithmetic types? How about std::uninitialized_fill?
  // do {
  //   alloc_traits::construct(this->alloc_, this->end_, value);
  //   ++this->end_;
  //   --n;
  // } while (n > 0);
  this->end_ = this->begin_ + n;
  std::fill(this->begin_, this->end_, value);
}

template<typename T, typename Alloc>
template<typename ForwardIter>
typename
std::enable_if<internal::is_forward_iterator<ForwardIter>::value, void>::type
matrix<T, Alloc>::construct_at_end_(ForwardIter first, ForwardIter last) {
  // TODO(Linh): Can we replace the do-while loop with std::copy since we're
  // only dealing with arithmetic types? How about std::uninitialized_copy?
  // for (; first != last; ++first, ++this->end_) {
  //   alloc_traits::construct(this->alloc_, this->end_, *first);
  // }
  // Must be this->end_ for the third argument!!!
  this->end_ = std::copy(first, last, this->end_);
}

// Replaces the contents of the buffer with that in the range [first, last).
// Memory allocated if neccessary.
template<typename T, typename Alloc>
template<typename ForwardIter>
typename
std::enable_if<
  internal::is_forward_iterator<ForwardIter>::value &&
  std::is_constructible<
    T,
    typename std::iterator_traits<ForwardIter>::reference>::value,
  void>::type
matrix<T, Alloc>::assign_(ForwardIter first, ForwardIter last) {
  size_type new_size = static_cast<size_type>(std::distance(first, last));
  if (new_size <= capacity()) {
    // ForwardIter mid = last;
    // bool growing = false;
    // if (new_size > size()) {
    //   growing = true;
    //   mid = first;
    //   std::advance(mid, size());
    // }
    // pointer could_be_new_end = std::copy(first, mid, this->begin_);
    // if (growing) {
    //   construct_at_end_(mid, last);
    // } else {
    //   this->destruct_at_end_(could_be_new_end);
    // }
    this->end_ = std::copy(first, last, this->begin_);
  } else {
    deallocate_memory_();
    allocate_memory_(new_size);
    construct_at_end_(first, last);
  }
}

template<typename T, typename Alloc>
void
matrix<T, Alloc>::move_assign_(matrix& m, std::true_type)  // NOLINT
    INSIGHT_NOEXCEPT_IF(
        std::is_nothrow_move_assignable<allocator_type>::value) {
  deallocate_memory_();
  base::move_assign_alloc_(m);
  this->begin_ = m.begin_;
  this->end_ = m.end_;
  this->end_cap_ = m.end_cap_;
  this->dim_ = m.dim_;  // TODO(Linh): or std::move(m.dim_)
  m.begin_ = m.end_ = m.end_cap_ = nullptr;
  m.dim_ = std::make_pair(0, 0);  // TODO(Linh): is it neccessary?
}

// TODO(Linh): This will be nothrow if alloc_traits::is_always_equal is true
// but this feature is only available since C++17.
template<typename T, typename Alloc>
void
matrix<T, Alloc>::move_assign_(matrix& m, std::false_type) {  // NOLINT
  // when propagate_on_container_move_assignment is false allocator will be
  // kept (no replacement happens) therefore we need to check to see wehther
  // this's allocator and m's allocator are in deed the same.
  if (this->alloc_ != m.alloc_) {
    using MoveIter = std::move_iterator<iterator>;
    // This can throw!
    assign_(MoveIter(m.begin()), MoveIter(m.end()));
    dim_ = m.dim_;
  } else {
    move_assign_(m, std::true_type());
  }
}

// In addition to the public member swap, we also need a free swap function.
template<typename T, typename Alloc>
inline
void swap(matrix<T, Alloc>& m1, matrix<T, Alloc>& m2) INSIGHT_NOEXCEPT_IF(
    noexcept(m1.swap(m2))) {
  m1.swap(m2);
}


// matrix-scalar arithmetic.

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::mul_scalar_(const_reference scalar, std::true_type) {
  linalg_detail::blas_scal(size(), scalar, this->begin_);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::mul_scalar_(const_reference scalar, std::false_type) {
  std::for_each(this->begin_, this->end_, [&](reference e) { e *= scalar; });
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::div_scalar_(const_reference scalar, std::true_type) {
  linalg_detail::blas_scal(size(), value_type(1.0) / scalar, this->begin_);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::div_scalar_(const_reference scalar, std::false_type) {
  std::for_each(this->begin_, this->end_, [&](reference e) { e /= scalar; });
}

// matrix-matrix arithmetic.

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::add_matrix_(const matrix& m, std::true_type) {
  linalg_detail::blas_add(size(), m.data(), this->begin_, this->begin_);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::add_matrix_(const matrix& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e += *it++; });
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::sub_matrix_(const matrix& m, std::true_type) {
  linalg_detail::blas_sub(size(), this->begin_, m.data(), this->begin_);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::sub_matrix_(const matrix& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e -= *it++; });
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::mul_matrix_(const matrix& m, std::true_type) {
  linalg_detail::blas_mul(size(), m.data(), this->begin_, this->begin_);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::mul_matrix_(const matrix& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e *= *it++; });
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::div_matrix_(const matrix& m, std::true_type) {
  linalg_detail::blas_div(size(), this->begin_, m.data(), this->begin_);
}

template<typename T, typename Alloc>
inline
void
matrix<T, Alloc>::div_matrix_(const matrix& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e /= *it++; });
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_MATRIX_H_
