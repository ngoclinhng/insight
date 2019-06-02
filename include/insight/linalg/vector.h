// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_VECTOR_H_
#define INCLUDE_INSIGHT_LINALG_VECTOR_H_

#include <cmath>
#include <utility>
#include <initializer_list>
#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <numeric>

#include "insight/internal/storage.h"
#include "insight/internal/math_functions.h"
#include "insight/linalg/arithmetic_expression.h"
#include "insight/linalg/evaluator.h"
#include "glog/logging.h"

namespace insight {

// Dense, column vector.
template<typename T, typename Allocator = insight_allocator<T> >
class vector: public vector_expression<vector<T, Allocator> >,
              private internal::storage<T, Allocator> {
  using self_type = vector<T, Allocator>;  // NOLINT
  using buffer = internal::storage<T, Allocator>;

 public:
  // public types.
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = typename std::allocator_traits<Allocator>::size_type;
  using difference_type =
      typename std::allocator_traits<Allocator>::difference_type;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer =
      typename std::allocator_traits<Allocator>::const_pointer;

  using iterator = pointer;
  using const_iterator = const_pointer;

  using shape_type = std::pair<size_type, size_type>;

  static constexpr bool is_vector = true;

  // Constructs an empty dense vector.
  vector() : buffer() {}

  // Constructs a vector of size `count`, all elements are initialized to
  // `value` (which is `zero` by default). Use the optional argument `alloc`
  // to customize memory management.
  vector(size_type count, const_reference value = value_type(),
         const allocator_type& alloc = insight_allocator<T>())
      : buffer(count, alloc) {
    // TODO(Linh): Should we use std::fill instead for trivially constructible
    // type T?
    std::uninitialized_fill_n(buffer::start, count, value);
  }

  // Copy constructor.
  vector(const vector& v) : buffer(v.size(), v.alloc) {
    // TODO(Linh): Should we use std::copy instead?
    std::uninitialized_copy_n(v.begin(), v.size(), buffer::start);
  }

  // assignment operator.
  self_type& operator=(const self_type& rhs) {
    // self-assignment
    if (this == &rhs) { return *this; }

    // Only allocate new memory when the current capacity is less than the
    // size of the right hand side vector `rhs`.
    if (capacity()  < rhs.size()) {
      self_type temp(rhs);
      temp.swap(*this);
      return *this;
    }

    // Otherwise, reuse memory and copy data over.
    buffer::alloc = rhs.alloc;
    copy_data(rhs.begin(), rhs.size());
    buffer::end = buffer::start + rhs.size();
    return *this;
  }

  // Move constructor.
  vector(vector&& temp) noexcept : buffer() {
    temp.swap(*this);
  }

  // Move assignment operator.
  self_type& operator=(self_type&& rhs) noexcept {
    if (this == &rhs) { return *this; }
    rhs.swap(*this);
    return *this;
  }

  // Destructor.
  ~vector() { destroy_elements(); }

  // Constructs a vector from the input iterator range [first, last).
  template<typename InputIt>
  vector(InputIt first, InputIt last,
         const allocator_type& alloc = Allocator())
      : buffer(std::distance(first, last), alloc) {
    std::uninitialized_copy_n(first, std::distance(first, last),
                              buffer::start);
  }

  // Constructs a vector with the contents of the initializer list `init`.
  vector(const std::initializer_list<T>& init,
         const allocator_type& alloc = insight_allocator<T>())
      : buffer(init.size(), alloc) {
    std::uninitialized_copy_n(init.begin(), init.size(), buffer::start);
  }

  // Assigns to an initializer list `init`.
  self_type& operator=(const std::initializer_list<T>& init) {
    if (capacity() < init.size()) {
      self_type temp(init);
      temp.swap(*this);
      return *this;
    }

    copy_data(init.begin(), init.size());
    buffer::end = buffer::start + init.size();
    return *this;
  }

  // Constructs a vector from a `normal` vector expression.
  template<typename E>
  vector(const vector_expression<E>& expr,
         const allocator_type& alloc = Allocator())  // NOLINT
      : buffer(expr.self().size(), alloc) {
    evaluator<E>::assign(expr.self(), buffer::start);
  }

  // Assigns to a `normal` vector expression.
  template<typename E>
  self_type& operator=(const vector_expression<E>& expr) {
    if (capacity() < expr.size()) {
      self_type temp(expr);
      temp.swap(*this);
      return *this;
    }

    evaluator<E>::assign(expr.self(), buffer::start);
    buffer::end = buffer::start + expr.self().size();
    return *this;
  }

  // Returns the number of rows in `this` vector.
  inline size_type num_rows() const {return size(); }

  // Returns the number of columns in this vector.
  inline size_type num_cols() const {return (size() == 0) ? 0 : 1; }

  // Returns the shape of this vector.
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }

  // Returns the number of elements in `this` vector.
  inline size_type size() const { return (buffer::end - buffer::start); }

  // Returns true if this vector is empty.
  inline bool empty() const { return size() == 0; }

  // Returns the number of elements that the vector has currently allocated
  // space for.
  size_type capacity() const { return (buffer::last - buffer::start); }

  // Returns the allocator.
  allocator_type get_allocator() const { return buffer::alloc; }

  // Accesses the element at index `i` in the vector without bounce-checking.
  // A const reference is returned.
  inline const_reference operator[](size_type i) const {
    return buffer::start[i];
  }

  // Accesses the element at index `i` in the vector without bounce-checking.
  // A reference is returned.
  inline reference operator()(size_type i) const {
    return buffer::start[i];
  }

  // Element-wise iterator

  iterator begin() { return buffer::start; }
  const_iterator begin() const { return buffer::start; }
  const_iterator cbegin() const { return buffer::start; }

  iterator end() { return buffer::end; }
  const_iterator end() const { return buffer::end; }
  const_iterator cend() const { return buffer::end; }

  // Swap two vectors.
  void swap(self_type& other) noexcept {
    using std::swap;
    swap(static_cast<buffer&>(*this), static_cast<buffer&>(other));
  }

  // vector-scalar arithmetic.

  // Increments each and every element in the vector by the constant
  // `scalar`.
  inline self_type& operator+=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e += scalar; });
    return *this;
  }

  // Decrements each and every element in the vector by the constant
  // `scalar`.
  inline self_type& operator-=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e -= scalar; });
    return *this;
  }

  // Replaces each and every element in the vector by the result of
  // multiplication of that element and a `scalar`.
  inline self_type& operator*=(value_type scalar) {
    if (std::is_floating_point<T>::value) {
      internal::insight_scal(size(), scalar, begin());
    } else {
      std::for_each(begin(), end(), [&](reference e) { e *= scalar; });
    }
    return *this;
  }

  // Replaces each and every element in the vector by the result of
  // dividing that element by a constant `scalar`.
  inline self_type& operator/=(value_type scalar) {
    if (std::is_floating_point<T>::value) {
      internal::insight_scal(size(), value_type(1.0) / scalar, begin());
    } else {
      std::for_each(begin(), end(), [&](reference e) { e /= scalar; });
    }
    return *this;
  }

  // vector-vector arithmetic.

  // Replaces each and every element in `this` vector by the result of
  // adding that element with the corresponfing element in `other`
  // vector.
  inline self_type& operator+=(const vector& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately by multiplying `this` by `2` for example?
    CHECK_EQ(size(), other.size());
    if (std::is_floating_point<T>::value) {
      internal::insight_add(size(), other.begin(), begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e += *it++; });
    }
    return *this;
  }

  // Replaces each and every element in `this` vector by the result of
  // substracting the corresponding element in `other` vector from that
  // element.
  inline self_type& operator-=(const vector& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately by setting all elements of `this` to `zero`?
    CHECK_EQ(size(), other.size());
    if (std::is_floating_point<T>::value) {
      internal::insight_sub(size(), begin(), other.begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e -= *it++; });
    }
    return *this;
  }

  // Replaces each and every element in `this` vector by the result of
  // multiplying that element and the corresponding element in the
  // `other` vector.
  inline self_type& operator*=(const vector& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately?
    CHECK_EQ(size(), other.size());
    if (std::is_floating_point<T>::value) {
      internal::insight_mul(size(), other.begin(), begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e *= *it++; });
    }
    return *this;
  }

  // Replaces each and every element in `this` vector by the result of
  // dividing that element by the corresponding element in the `other`
  // vector.
  inline self_type& operator/=(const vector& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately by setting all elements of `this` to `one`?
    CHECK_EQ(size(), other.size());
    if (std::is_floating_point<T>::value) {
      internal::insight_div(size(), begin(), other.begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e /= *it++; });
    }
    return *this;
  }

  // Returns the l2 norm of `this` vector
  inline value_type norm2() const {
    if (std::is_floating_point<T>::value) {
      return internal::insight_nrm2(size(), begin());
    } else {
      auto op = [](const_reference partial, const_reference val) {
                  return partial + val * val;
                };
      value_type sum_of_squares =
          std::accumulate(begin(), end(), value_type(0.0), op);
      return std::sqrt(sum_of_squares);
    }
  }

  // Returns the dot product of `this` vector and the `other` vector.
  inline value_type dot(const vector& other) const {
    CHECK_EQ(size(), other.size());
    if (std::is_floating_point<T>::value) {
      return internal::insight_dot(size(), begin(), other.begin());
    } else {
      return std::inner_product(begin(), end(), other.begin(),
                                value_type(0.0));
    }
  }

  // Vector expresison arithmetic.

  template<typename E>
  inline self_type& operator+=(const vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    evaluator<E>::add(expr.self(), buffer::start);
    return *this;
  }

  template<typename E>
  inline self_type& operator-=(const vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    evaluator<E>::sub(expr.self(), buffer::start);
    return *this;
  }

  template<typename E>
  inline self_type& operator*=(const vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    evaluator<E>::mul(expr.self(), buffer::start);
    return *this;
  }

  template<typename E>
  inline self_type& operator/=(const vector_expression<E>& expr) {
    CHECK_EQ(size(), expr.self().size());
    evaluator<E>::div(expr.self(), buffer::start);
    return *this;
  }

 private:
  void destroy_elements() {
    // TODO(Linh): Do we really need to do this for trivially destructible
    // type T?
    pointer p = buffer::start;
    while (p != buffer::end) { p->~value_type(); ++p; }
  }

  template<typename InputIt>
  void copy_data(InputIt first, size_type count) {
    size_type sz = size();

    if (count <= sz) {
      // Copy over old elements
      std::copy(first, first + count, buffer::start);

      // Destroy surplus elements.
      pointer p = buffer::start + count;
      while (p != buffer::end) { p->~value_type(); ++p; }
    } else {
      // Copy over old elements
      std::copy(first, first + sz, buffer::start);

      // Construct additional elements.
      std::uninitialized_copy_n(first + sz, count - sz, buffer::end);
    }
  }
};

// According to the Effective C++ swap idiom (item 25), we also need to
// provide a free swap function.
template<typename T, typename Allocator>
void swap(vector<T, Allocator>& v1,
          vector<T, Allocator>& v2) {  // NOLINT
  v1.swap(v2);
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_VECTOR_H_
