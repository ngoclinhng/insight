// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_VECTOR_H_
#define INCLUDE_INSIGHT_LINALG_VECTOR_H_

#include <algorithm>
#include <limits>
#include <utility>
#include <initializer_list>
#include <numeric>
#include <cmath>

#include "insight/memory.h"

#include "insight/linalg/detail/expression_evaluator.h"
#include "insight/linalg/detail//dense_base.h"
#include "insight/linalg/detail/blas_routines.h"

#include "glog/logging.h"

namespace insight {

// Dense column vector.
template<typename T, typename Alloc = allocator<T> >  // NOLINT
class vector
    : private linalg_detail::dense_base<T, Alloc>,
      public linalg_detail::vector_expression<vector<T, Alloc> > {
 private:
  using base = linalg_detail::dense_base<T, Alloc>;
  using self = vector;
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
                "vector<T> only accepts arithmetic types.");

  // Constructs an empty vector. An empty vector is a vector with zero
  // number of elements.
  vector() INSIGHT_NOEXCEPT_IF(
      std::is_nothrow_default_constructible<allocator_type>::value) {
  }

  // Constructs a dense vector with n number of elemetns, all elements are
  // default-constructed. If n == 0 then an empty vector will be constructed.
  explicit vector(size_type n);

  // Constructs a dense vector with n number of elements, all elements are
  // copy-constructed from value. If n == 0 then an empty vector will be
  // constructed.
  vector(size_type n, const value_type& value);

  // Constructs a vector with the contents in the range [first, last).
  // If std::distance(first, last) == 0, then an empty vector will be
  // constructed.
  template<typename ForwardIter>
  vector(ForwardIter first, ForwardIter last,
         typename std::enable_if<
         internal::is_forward_iterator<ForwardIter>::value &&
         std::is_constructible<value_type,
         typename std::iterator_traits<ForwardIter>::reference>::value
         >::type* = 0);

  // TODO(Linh): Should this be default, i.e = default instead?
  ~vector() {}

  // Copy constructor and assignment operator.
  vector(const vector& m);
  vector& operator=(const vector& m);

  // Move constructor & move assignment operator.
  vector(vector&& m) INSIGHT_NOEXCEPT_IF(
      std::is_nothrow_move_constructible<allocator_type>::value);
  vector& operator=(vector&& m) INSIGHT_NOEXCEPT_IF(
      alloc_traits::propagate_on_container_move_assignment::value &&
      std::is_nothrow_move_assignable<allocator_type>::value);

  // Constructs a vector with the contents of the initializer_list il
  // If il.size() == 0, then an empty vector will be contructed.
  vector(std::initializer_list<value_type> il);
  vector& operator=(std::initializer_list<value_type> il);

  // Constructs a vector from a generic vector expression.

  template<typename E>
  vector(const linalg_detail::vector_expression<E>& expr);  // NOLINT

  template<typename E>
  vector& operator=(const linalg_detail::vector_expression<E>& expr);

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

  inline size_type row_count() const INSIGHT_NOEXCEPT { return size(); }
  inline size_type col_count() const INSIGHT_NOEXCEPT {
    return size() > 0 ? 1 : 0;
  }
  inline const shape_type shape() const INSIGHT_NOEXCEPT {
    return std::make_pair(row_count(), col_count());
  }

  // Returns the number of elements in the vector.
  inline size_type size() const INSIGHT_NOEXCEPT { return base::size(); }

  // Returns the number of elements that the container has allocated space
  // for
  inline size_type capacity() const INSIGHT_NOEXCEPT {
    return base::capacity();
  }

  // Returns true if the vector is empty, i.e there is no elements in the
  // vector, and row_count() == col_count() == 0.
  inline bool empty() const INSIGHT_NOEXCEPT {
    return (this->begin_ == this->end_);
  }

  // Returns the maximum number of elements the vector is able to hold.
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

  // Returns the pointer to the underlying array.
  inline value_type* data() INSIGHT_NOEXCEPT { return this->begin_; }
  inline const value_type* data() const INSIGHT_NOEXCEPT {
    return this->begin_;
  }

  // Returns the transpose of this vector.
  inline linalg_detail::transpose_expression<self> t() const {
    return linalg_detail::transpose_expression<self>(*this);
  }

  // Clear all the contents in the vector and set its size to zero.
  // Vector will become emtpy after clear call.
  void clear() INSIGHT_NOEXCEPT { base::clear(); }

  // void reshape(size_type new_row_count, size_type new_col_count)
  //     INSIGHT_NOEXCEPT;

  void swap(vector& m) INSIGHT_NOEXCEPT_IF(
      !alloc_traits::propagate_on_container_swap::value ||
      internal::is_nothrow_swappable<allocator_type>::value);

  // Vector-scalar arithmetic.

  // Increments each and every element in the vector by the constant scalar.
  inline vector& operator+=(const value_type& scalar) {
    std::for_each(this->begin_, this->end_, [&](reference e) { e += scalar; });
    return *this;
  }

  // Decrements each and every element in the vector by the constant scalar.
  inline vector& operator-=(const value_type& scalar) {
    std::for_each(this->begin_, this->end_, [&](reference e) { e -= scalar; });
    return *this;
  }

  // Replaces each and every element in the vector by the result of
  // multiplication of that element and a scalar.
  inline vector& operator*=(const value_type& scalar) {
    mul_scalar_(scalar, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in the vector by the result of
  // dividing that element by a constant scalar.
  inline vector& operator/=(const value_type& scalar) {
    div_scalar_(scalar, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // vector-vector arithmetic.

  // Replaces each and every element in this vector by the result of
  // adding that element with the corresponfing element in the vector m.
  inline vector& operator+=(const vector& m) {
    CHECK_EQ(size(), m.size());
    add_vector_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in this vector by the result of
  // substracting the corresponding element in the vector m from that
  // element.
  inline vector& operator-=(const vector& m) {
    CHECK_EQ(size(), m.size());
    sub_vector_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in this vector by the result of
  // multiplying that element and the corresponding element in the vector m.
  inline vector& operator*=(const vector& m) {
    CHECK_EQ(size(), m.size());
    mul_vector_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // Replaces each and every element in this vector by the result of
  // dividing that element by the corresponding element in the vector m.
  inline vector& operator/=(const vector& m) {
    CHECK_EQ(size(), m.size());
    div_vector_(m, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
    return *this;
  }

  // vector expression arithmetic.

  template<typename E>
  inline vector& operator+=(const linalg_detail::vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.add(this->begin_);
    return *this;
  }

  template<typename E>
  inline vector& operator-=(const linalg_detail::vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.sub(this->begin_);
    return *this;
  }

  template<typename E>
  inline vector& operator*=(const linalg_detail::vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.mul(this->begin_);
    return *this;
  }

  template<typename E>
  inline vector& operator/=(const linalg_detail::vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.div(this->begin_);
    return *this;
  }

  // Returns the L2-norm of this vector.
  inline value_type nrm2() const {
    return nrm2_(std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
  }

  // Returns the dot product of this vector with the vector m.
  inline value_type dot(const vector& v) const {
    CHECK_EQ(size(), v.size());
    return dot_(v, std::integral_constant<bool, std::is_floating_point<value_type>::value>());  // NOLINT
  }

 private:
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

  // Move m to this vector when propagate_on_container_move_assignment
  // is true. this's allocator will be replaced
  void move_assign_(vector& m, std::true_type) INSIGHT_NOEXCEPT_IF(  // NOLINT
      std::is_nothrow_move_assignable<allocator_type>::value);

  // Move m to this vector whenpropagate_on_container_move_assignment
  // is false. the old allocator is kept.
  void move_assign_(vector& m, std::false_type);  // NOLINT

  // helper methods for vector-scalar arithmetic.

  void mul_scalar_(const value_type& scalar, std::true_type);
  void mul_scalar_(const value_type& scalar, std::false_type);

  void div_scalar_(const value_type& scalar, std::true_type);
  void div_scalar_(const value_type& scalar, std::false_type);

  // Helper methods for vector-vector arithmetic.

  void add_vector_(const vector& m, std::true_type);
  void add_vector_(const vector& m, std::false_type);

  void sub_vector_(const vector& m, std::true_type);
  void sub_vector_(const vector& m, std::false_type);

  void mul_vector_(const vector& m, std::true_type);
  void mul_vector_(const vector& m, std::false_type);

  void div_vector_(const vector& m, std::true_type);
  void div_vector_(const vector& m, std::false_type);

  // L2-norm helpers
  value_type nrm2_(std::true_type) const;
  value_type nrm2_(std::false_type) const;

  // Dot helpers
  value_type dot_(const vector& v, std::true_type) const;
  value_type dot_(const vector& v, std::false_type) const;
};

template<typename T, typename Alloc>
vector<T, Alloc>::vector(size_type n) {
  if (n > 0) {
    allocate_memory_(n);
    construct_at_end_(n);
  }
}

template<typename T, typename Alloc>
vector<T, Alloc>::vector(size_type n, const value_type& value) {
  if (n > 0) {
    allocate_memory_(n);
    construct_at_end_(n, value);
  }
}

template<typename T, typename Alloc>
template<typename ForwardIter>
vector<T, Alloc>::vector(ForwardIter first, ForwardIter last,
                         typename
                         std::enable_if<
                         internal::is_forward_iterator<ForwardIter>::value &&
                         std::is_constructible<value_type,
                         typename std::iterator_traits<ForwardIter>::reference>::value>::type*) {
  size_type n = static_cast<size_type>(std::distance(first, last));
  if (n > 0) {
    allocate_memory_(n);
    construct_at_end_(first, last);
  }
}

template<typename T, typename Alloc>
vector<T, Alloc>::vector(const vector& m)
    : base(alloc_traits::select_on_container_copy_construction(m.alloc_)) {
  size_type n = m.size();
  if (n > 0) {
    allocate_memory_(n);
    construct_at_end_(m.begin_, m.end_);
  }
}

template<typename T, typename Alloc>
inline
vector<T, Alloc>&
vector<T, Alloc>::operator=(const vector& m) {
  if (this != &m) {
    base::copy_assign_alloc_(m);
    assign_(m.begin_, m.end_);
  }
  return *this;
}

template<typename T, typename Alloc>
inline
vector<T, Alloc>::vector(vector&& m) INSIGHT_NOEXCEPT_IF(
    std::is_nothrow_move_constructible<allocator_type>::value)
    :   base(std::move(m.alloc_)) {
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
vector<T, Alloc>&
vector<T, Alloc>::operator=(vector&& m) INSIGHT_NOEXCEPT_IF(
    alloc_traits::propagate_on_container_move_assignment::value &&
    std::is_nothrow_move_assignable<allocator_type>::value) {
  move_assign_(m, std::integral_constant<bool, alloc_traits::propagate_on_container_move_assignment::value>());  // NOLINT
  return *this;
}

template<typename T, typename Alloc>
inline
vector<T, Alloc>::vector(std::initializer_list<value_type> il) {
  size_type n = il.size();
  if (n > 0) {
    allocate_memory_(n);
    construct_at_end_(il.begin(), il.end());
  }
}

template<typename T, typename Alloc>
inline
vector<T, Alloc>&
vector<T, Alloc>::operator=(std::initializer_list<value_type> il) {
  assign_(il.begin(), il.end());
  return *this;
}

template<typename T, typename Alloc>
template<typename E>
vector<T, Alloc>::vector(const linalg_detail::vector_expression<E>& expr) {
  size_type n = expr.self().size();
  if (n > 0) {
    allocate_memory_(n);
    linalg_detail::expression_evaluator<E> evaluator(expr.self());
    evaluator.assign(this->begin_);
    this->end_ = this->begin_ + n;
  }
}

template<typename T, typename Alloc>
template<typename E>
vector<T, Alloc>&
vector<T, Alloc>::operator=(const linalg_detail::vector_expression<E>& expr) {
  size_type new_size = expr.self().size();
  if (new_size > capacity()) {
    deallocate_memory_();
    allocate_memory_(new_size);
  }
  linalg_detail::expression_evaluator<E> evaluator(expr.self());
  evaluator.assign(this->begin_);
  this->end_ = this->begin_ + new_size;
  return *this;
}

template<typename T, typename Alloc>
typename vector<T, Alloc>::size_type
vector<T, Alloc>::max_size() const INSIGHT_NOEXCEPT {
  return std::min<size_type>(alloc_traits::max_size(this->alloc_),
                             std::numeric_limits<difference_type>::max());
}

template<typename T, typename Alloc>
void
vector<T, Alloc>::swap(vector& m) INSIGHT_NOEXCEPT_IF(
    !alloc_traits::propagate_on_container_swap::value ||
    internal::is_nothrow_swappable<allocator_type>::value) {
  DCHECK(alloc_traits::propagate_on_container_swap::value ||
         (this->alloc_ == m.alloc_))
      << "vector::swap Either propagate_on_container_swap must be true "
      << "or the allocators must compare equal";
  using std::swap;
  swap(this->begin_, m.begin_);
  swap(this->end_, m.end_);
  swap(this->end_cap_, m.end_cap_);
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
vector<T, Alloc>::allocate_memory_(size_type n) {
  // TODO(Linh): throw or CHECK?
  if (n > max_size())
    this->throw_length_error("vector::allocate_memory_(n): "
                             "the requested size n is too large");
  this->begin_ = this->end_ = alloc_traits::allocate(this->alloc_, n);
  this->end_cap_ = this->begin_ + n;
}

// Deallocate memory.
template<typename T, typename Alloc>
void
vector<T, Alloc>::deallocate_memory_() INSIGHT_NOEXCEPT {
  if (this->begin_ != nullptr) {
    // TODO(Linh): Can we just skip the clear step since we're only dealing
    // with arithmetic types?
    clear();
    alloc_traits::deallocate(this->alloc_, this->begin_, capacity());
    this->begin_ = this->end_ = this->end_cap_ = nullptr;
  }
}

// Default constructs n objects starting at end_.
// throws if constructions throws.
// Precondition: n > 0
// Precondition: size() + n <= capacity()
// Postcondition: size() == size() + n
template<typename T, typename Alloc>
void
vector<T, Alloc>::construct_at_end_(size_type n) {
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
vector<T, Alloc>::construct_at_end_(size_type n, const_reference value) {
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
vector<T, Alloc>::construct_at_end_(ForwardIter first, ForwardIter last) {
  // TODO(Linh): Can we replace the do-while loop with std::copy since we're
  // only dealing with arithmetic types? How about std::uninitialized_copy?
  // for (; first != last; ++first, ++this->end_) {
  //   alloc_traits::construct(this->alloc_, this->end_, *first);
  // }
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
vector<T, Alloc>::assign_(ForwardIter first, ForwardIter last) {
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
vector<T, Alloc>::move_assign_(vector& m, std::true_type)  // NOLINT
    INSIGHT_NOEXCEPT_IF(
        std::is_nothrow_move_assignable<allocator_type>::value) {
  deallocate_memory_();
  base::move_assign_alloc_(m);
  this->begin_ = m.begin_;
  this->end_ = m.end_;
  this->end_cap_ = m.end_cap_;
  m.begin_ = m.end_ = m.end_cap_ = nullptr;
}

// TODO(Linh): This will be nothrow if alloc_traits::is_always_equal is true
// but this feature is only available since C++17.
template<typename T, typename Alloc>
void
vector<T, Alloc>::move_assign_(vector& m, std::false_type) {  // NOLINT
  // when propagate_on_container_move_assignment is false allocator will be
  // kept (no replacement happens) therefore we need to check to see wehther
  // this's allocator and m's allocator are in deed the same.
  if (this->alloc_ != m.alloc_) {
    using MoveIter = std::move_iterator<iterator>;
    // This can throw!
    assign_(MoveIter(m.begin()), MoveIter(m.end()));
  } else {
    move_assign_(m, std::true_type());
  }
}

// In addition to the public member swap, we also need a free swap function.
template<typename T, typename Alloc>
inline
void swap(vector<T, Alloc>& m1, vector<T, Alloc>& m2) INSIGHT_NOEXCEPT_IF(
    noexcept(m1.swap(m2))) {
  m1.swap(m2);
}


// vector-scalar arithmetic.

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::mul_scalar_(const value_type& scalar, std::true_type) {
  linalg_detail::blas_scal(size(), scalar, this->begin_);
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::mul_scalar_(const value_type& scalar, std::false_type) {
  std::for_each(this->begin_, this->end_, [&](reference e) { e *= scalar; });
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::div_scalar_(const value_type& scalar, std::true_type) {
  linalg_detail::blas_scal(size(), value_type(1.0) / scalar, this->begin_);
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::div_scalar_(const value_type& scalar, std::false_type) {
  std::for_each(this->begin_, this->end_, [&](reference e) { e /= scalar; });
}

// vector-vector arithmetic.

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::add_vector_(const vector& m, std::true_type) {
  linalg_detail::blas_add(size(), m.data(), this->begin_, this->begin_);
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::add_vector_(const vector& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e += *it++; });
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::sub_vector_(const vector& m, std::true_type) {
  linalg_detail::blas_sub(size(), this->begin_, m.data(), this->begin_);
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::sub_vector_(const vector& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e -= *it++; });
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::mul_vector_(const vector& m, std::true_type) {
  linalg_detail::blas_mul(size(), m.data(), this->begin_, this->begin_);
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::mul_vector_(const vector& m, std::false_type) {
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e *= *it++; });
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::div_vector_(const vector& m, std::true_type) {
  linalg_detail::blas_div(size(), this->begin_, m.data(), this->begin_);
}

template<typename T, typename Alloc>
inline
void
vector<T, Alloc>::div_vector_(const vector& m, std::false_type) {  // NOLINT
  auto it = m.begin();
  std::for_each(this->begin_, this->end_, [&](reference e) { e /= *it++; });
}

template<typename T, typename Alloc>
inline
T vector<T, Alloc>::nrm2_(std::true_type) const {
  return  linalg_detail::blas_nrm2(size(), this->begin_);
}

template<typename T, typename Alloc>
T vector<T, Alloc>::nrm2_(std::false_type) const {
  auto op = [](const_reference partial, const_reference val) {
              return partial + val * val;
            };
  value_type sum_of_squares =
      std::accumulate(this->begin_, this->end_, value_type(0.0), op);
  return std::sqrt(sum_of_squares);
}

template<typename T, typename Alloc>
inline
T
vector<T, Alloc>::dot_(const vector& v, std::true_type) const {
  return  linalg_detail::blas_dot(size(), this->begin_, v.data());
}

template<typename T, typename Alloc>
inline
T
vector<T, Alloc>::dot_(const vector& v, std::false_type) const {
  return std::inner_product(begin(), end(), v.begin(), value_type(0.0));
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_VECTOR_H_
