// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_MATRIX_H_
#define INCLUDE_INSIGHT_LINALG_MATRIX_H_

#include <cstdlib>
#include <utility>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <initializer_list>
#include <iterator>

#include "insight/internal/storage.h"
#include "insight/internal/math_functions.h"
#include "insight/linalg/arithmetic_expression.h"
#include "insight/linalg/evaluator.h"

#include "glog/logging.h"

namespace insight {

template<typename T, typename Allocator = insight_allocator<T> >
class matrix:
      public matrix_expression<matrix<T, Allocator> >,
      private internal::storage<T, Allocator> {
  using self_type = matrix<T, Allocator>;
  using buffer = internal::storage<T, Allocator>;

 public:
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

  static constexpr bool is_vector = false;

  // Constructs an empty dense matrix.
  matrix() : buffer(),
             num_rows_(0),
             num_cols_(0) {
  }

  // Constructs a dense matrix of size `(num_rows, num_cols)`, all elements
  // are initialized to `value` (which is `zero` by default). Use the
  // optional `alloc` argument to customize the memory management.
  matrix(size_type num_rows,
         size_type num_cols,
         const_reference value = value_type(),
         const allocator_type& alloc = insight_allocator<T>())
      : buffer(num_rows * num_cols, alloc),
        num_rows_(num_rows),
        num_cols_(num_cols) {
    // TODO(Linh): Should we use std::fill instead for trivially
    // constructible type T?
    std::uninitialized_fill_n(buffer::start, num_rows * num_cols, value);
  }

  // Constructs a dense matrix with the specified dimension `dim`, whose
  // number of rows is equal to `dim.first`, and number of columns is equal
  // to `dim.second`. all elements are initialized to `value`
  // (which is `zero` by default). Use the optional `alloc` argument to
  // customize the memory management.
  explicit matrix(shape_type dim,
                  const_reference value = value_type(),
                  const allocator_type& alloc = insight_allocator<T>())
      : buffer(dim.first * dim.second, alloc),
        num_rows_(dim.first),
        num_cols_(dim.second) {
    std::uninitialized_fill_n(buffer::start, dim.first * dim.second, value);
  }

  // copy constructor.
  matrix(const matrix& m)
      : buffer(m.size(), m.alloc),
        num_rows_(m.num_rows_),
        num_cols_(m.num_cols_) {
    // TODO(Linh): should we use std::copy for trivially constructible type
    // T?
    std::uninitialized_copy_n(m.begin(), m.size(), buffer::start);
  }

  // assignment operator.
  self_type& operator=(const self_type& rhs) {
    // self-assignment
    if (this == &rhs) { return *this; }

    // Only allocate new memory when the current capacity is less than
    // the size of the left hand side matrix.
    if (capacity() < rhs.size()) {
      self_type temp(rhs);
      temp.swap(*this);
      return *this;
    }

    // Otherwise, reuse memory and copy data over.
    buffer::alloc = rhs.alloc;
    copy_data(rhs.begin(), rhs.size());
    buffer::end = buffer::start + rhs.size();
    num_rows_ = rhs.num_rows_;
    num_cols_ = rhs.num_cols_;

    return *this;
  }

  // Move constructor.
  matrix(matrix&& temp) noexcept : buffer(),
                                   num_rows_(0),
                                   num_cols_(0) {
    temp.swap(*this);
  }

  // Move assignment operator.
  self_type& operator=(self_type&& rhs) noexcept {
    if (this == &rhs) { return *this; }
    rhs.swap(*this);
    return *this;
  }

  // Destructor.
  ~matrix() { destroy_elements(); }

  // Constructs a 1 by `n` matrix ( a row vector ) with the contents of the
  // initializer list `init` where `n` is the size of `init`.
  matrix(const std::initializer_list<T>& init,
         const allocator_type& alloc = insight_allocator<T>())
      : buffer(init.size(), alloc),
        num_rows_(init.size() > 0 ? 1 : 0),
        num_cols_(init.size()) {
    std::uninitialized_copy_n(init.begin(), init.size(), buffer::start);
  }

  // Assigns to an initializer list.
  self_type& operator=(const std::initializer_list<T>& init) {
    if (capacity() < init.size()) {
      self_type temp(init);
      temp.swap(*this);
      return *this;
    }

    copy_data(init.begin(), init.size());
    buffer::end = buffer::start + init.size();
    num_rows_ = init.size() > 0 ? 1 : 0;
    num_cols_ = init.size();

    return *this;
  }

  // Constructs a `m` by `n` matrix with the contents of the initializer
  // list `init`, where `m` is the size of `init` and `n` is the size of
  // each of its sub-list.
  matrix(const std::initializer_list<std::initializer_list<T>>& init,
         const allocator_type& alloc = insight_allocator<T>())
      : buffer(0, alloc),
        num_rows_(0),
        num_cols_(0) {
    if (init.size() == 0) { return; }

    // Make sure all sub-lists are of the same size.
    auto sublist_size = init.begin()->size();
    CHECK_GT(sublist_size, static_cast<size_type>(0));
    CHECK(std::all_of(init.begin(), init.end(),
                      [&](const std::initializer_list<T>& sublist) {
                        return (sublist.size() == sublist_size);
                      }))
        << "Invalid nested initializer list: all sublists must have the "
        << "same number of elements";

    // Allocate memory.
    buffer::reserve(init.size() * sublist_size);

    // Copy data over.
    pointer p = buffer::start;
    for (auto it = init.begin(); it != init.end(); ++it) {
      std::uninitialized_copy_n(it->begin(), sublist_size, p);
      p += sublist_size;
    }

    buffer::end = buffer::start + init.size() * sublist_size;
    num_rows_ = init.size();
    num_cols_ = sublist_size;
  }

  // Assigns to a nested initializer list.
  self_type&
  operator=(const std::initializer_list<std::initializer_list<T>>& init) {
    CHECK_GT(init.size(), static_cast<size_type>(0));

    // Make sure all sub-lists are of the same size.
    auto sublist_size = init.begin()->size();
    CHECK_GT(sublist_size, static_cast<size_type>(0));
    CHECK(std::all_of(init.begin(), init.end(),
                      [&](const std::initializer_list<T>& sublist) {
                        return (sublist.size() == sublist_size);
                      }))
        << "Invalid nested initializer list: all sublists must have the "
        << "same number of elements";

    size_type count = init.size() * sublist_size;

    if (capacity() < count) {
      self_type temp(init);
      temp.swap(*this);
      return *this;
    }

    size_type this_size = size();

    if (this_size >= count) {
      pointer p = buffer::start;

      // Copy data over.
      for (auto it = init.begin(); it != init.end(); ++it) {
        std::copy(it->begin(), it->end(), p);
        p += sublist_size;
      }

      // Destroy surplus elements.
      while (p != buffer::end) { p->~value_type(); ++p; }
    } else if (this_size >= sublist_size) {
      // TODO(Linh): Is this too odd?
      std::ldiv_t dv = std::ldiv(this_size, sublist_size);
      size_type quot = static_cast<size_type>(dv.quot);
      size_type rem = static_cast<size_type>(dv.rem);

      pointer p = buffer::start;
      auto it = init.begin();
      size_type i = 0;

      // Copy data over.

      while (true) {
        if (i == quot) { break; }
        std::copy(it->begin(), it->end(), p);
        p += sublist_size;
        ++it;
        ++i;
      }

      std::copy(it->begin(), it->begin() + rem, p);
      p += rem;

      // Construct additional elements.

      std::uninitialized_copy_n(it->begin() + rem,
                                sublist_size - rem,
                                p);
      ++it;
      p += (sublist_size - rem);

      while (it != init.end()) {
        std::uninitialized_copy_n(it->begin(), sublist_size, p);
        p += sublist_size;
        ++it;
      }
    } else {
      auto it = init.begin();
      pointer p = buffer::start;
      std::copy(it->begin(), it->begin() + this_size, p);
      p += this_size;
      std::uninitialized_copy_n(it->begin() + this_size,
                                sublist_size - this_size,
                                p);
      p += (sublist_size - this_size);
      ++it;

      while (it != init.end()) {
        std::uninitialized_copy_n(it->begin(), sublist_size, p);
        p += sublist_size;
        ++it;
      }
    }

    buffer::end = buffer::start + count;
    num_rows_ = init.size();
    num_cols_ = sublist_size;

    return *this;
  }

  // constructs a `1` by `n` matrix from a given row_view where `n` is
  // the size of the row_view `view`.

  // class row_view;

  // matrix(const row_view& view,
  //        const allocator_type& alloc = insight_allocator<T>())  // NOLINT
  //     : buffer(view.size(), alloc),
  //       num_rows_(1),
  //       num_cols_(view.size()) {
  //   std::uninitialized_copy_n(view.begin(), view.size(), buffer::start);
  // }

  // self_type& operator=(const row_view& view) {
  //   if (capacity() < view.size()) {
  //     self_type temp(view);
  //     temp.swap(*this);
  //     return *this;
  //   }

  //   if (view.has_backing_matrix(*this) && (view.begin() == begin())) {
  //     // Destroy surplus elements.
  //     pointer p = buffer::start + view.size();
  //     while (p != buffer::end) { p->~value_type(); ++p; }
  //   } else {
  //     copy_data(view.begin(), view.size());
  //   }

  //   buffer::end = buffer::start + view.size();
  //   num_rows_ = 1;
  //   num_cols_ = view.size();

  //   return *this;
  // }

  // Constructs a matrix from a generic matrix expression.
  template<typename E>
  matrix(const matrix_expression<E>& expr,
         const allocator_type& alloc = Allocator())  // NOLINT
      : buffer(expr.self().size(), alloc),
        num_rows_(expr.self().num_rows()),
        num_cols_(expr.self().num_cols()) {
    evaluator<E>::assign(expr.self(), buffer::start);
  }

  // Assigns to a generic matrix expression.
  template<typename E>
  self_type& operator=(const matrix_expression<E>& expr) {
    if (capacity() < expr.self().size()) {
      self_type temp(expr);
      temp.swap(*this);
      return *this;
    }

    evaluator<E>::assign(expr.self(), buffer::start);
    buffer::end = buffer::start + expr.self().size();
    num_rows_ = expr.self().num_rows();
    num_cols_ = expr.self().num_cols();
    return *this;
  }

  // Return the number of rows in `this` matrix.
  inline size_type num_rows() const { return num_rows_; }

  // Returns the number of columns in `this` matrix.
  inline size_type num_cols() const { return num_cols_; }

  // Returns the `(num_rows, num_cols)` pair that represents the
  // shape/dimensions of `this` matrix.
  inline shape_type shape() const {
    return shape_type(num_rows_, num_cols_);
  }

  // Returns the number of elements in `this` matrix.
  inline size_type size() const { return (buffer::end - buffer::start); }

  // Returns true if this matrix is empty.
  inline bool empty() const { return size() == 0;}

  // Returns the number of elements that the matrix has currently
  // allocated space for.
  inline size_type capacity() const {
    return (buffer::last - buffer::start);
  }

  //
  // Returns the underlying allocator.
  inline allocator_type get_allocator() const { return buffer::alloc; }

  // Accesses the element at index i in the underlying buffer without
  // bounce-checking. A reference is returned.
  inline reference operator[](const size_type i) { return buffer::start[i]; }

  // Accesses the element at index i in the underlying buffer without
  // bounce-checking. A const reference is returned.
  inline const_reference operator[](const size_type i) const {
    return buffer::start[i];
  }

  // Accesses the element at row `i` and column `j` in the matrix without
  // bounce-checking. A reference is returned.
  inline reference operator()(const size_type i, const size_type j) {
    return buffer::start[i * num_cols_ + j];
  }

  // Accesses the element at row `i` and column `j` in the matrix without
  // bounce-checking. A const reference is returned.
  inline const_reference operator()(const size_type i,
                                    const size_type j) const {
    return buffer::start[i * num_cols_ + j];
  }

  // Element-wise iterator

  inline iterator begin() { return buffer::start; }
  inline const_iterator begin() const { return buffer::start; }
  inline const_iterator cbegin() const { return buffer::start; }

  inline iterator end() { return buffer::end; }
  inline const_iterator end() const { return buffer::end; }
  inline const_iterator cend() const { return buffer::end; }

  // Swap two matrices.
  void swap(self_type& other) noexcept {
    using std::swap;
    swap(static_cast<buffer&>(*this), static_cast<buffer&>(other));
    swap(num_rows_, other.num_rows_);
    swap(num_cols_, other.num_cols_);
  }


  // row view
  // ==================================================================

  class row_view {
    using matrix_type = matrix<T, Allocator>;
    using self_type = row_view;

   public:
    using value_type = typename matrix_type::value_type;
    using size_type = typename matrix_type::size_type;
    using difference_type = typename matrix_type::difference_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename matrix_type::pointer;
    using const_pointer = typename matrix_type::const_pointer;

    using shape_type = typename matrix_type::shape_type;

    row_view() = default;
    ~row_view() = default;

    row_view(matrix_type* base, size_type index)
        : backing_matrix_(base),
          row_start_(base->begin() + index * base->num_cols()),
          row_end_(base->begin() + (index + 1) * base->num_cols()) {
      CHECK_LT(index, base->num_rows()) << "row_view: invalid row index";
    }

    inline size_type num_rows() const { return 1; }

    inline size_type num_cols() const { return backing_matrix_->num_cols(); }

    inline shape_type shape() const {
      return shape_type(1, num_cols());
    }

    inline size_type size() const { return num_cols(); }

    inline bool empty() const { return size() == 0; }

    inline reference operator[](const size_type i) {
      return row_start_[i];
    }

    inline const_reference operator[](const size_type i) const {
      return row_start_[i];
    }

    // iterator.

    using iterator = pointer;
    using const_iterator = const_pointer;

    inline iterator begin() { return row_start_; }
    inline const_iterator begin() const { return row_start_; }
    inline const_iterator cbegin() const { return row_start_; }

    inline iterator end() { return row_end_; }
    inline const_iterator end() const { return row_end_; }
    inline const_iterator cend() const { return row_end_; }

    inline bool has_backing_matrix(const matrix_type& base) const {
      return (backing_matrix_ == &base);
    }

    // row-scalar arithmetic.

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
      if (std::is_floating_point<T>::value) {
        internal::insight_scal(size(), scalar, begin());
      } else {
        std::for_each(begin(), end(), [&](reference e) { e *= scalar; });
      }
      return *this;
    }

    // Replaces each and every element in the row by the result of
    // dividing that element by a constant `scalar`.
    inline self_type& operator/=(value_type scalar) {
      if (std::is_floating_point<T>::value) {
        internal::insight_scal(size(), value_type(1.0) / scalar, begin());
      } else {
        std::for_each(begin(), end(), [&](reference e) { e /= scalar; });
      }
      return *this;
    }

   private:
    matrix_type* backing_matrix_;
    pointer row_start_;
    pointer row_end_;
  };

  // Accesses the row at index `row_index`.
  inline row_view row_at(size_type row_index) {
    return row_view(this, row_index);
  }

  // matrix-scalar arithmetic.

  // Increments each and every element in the matrix by the constant
  // `scalar`.
  inline self_type& operator+=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e += scalar; });
    return *this;
  }

  // Decrements each and every element in the matrix by the constant
  // `scalar`.
  inline self_type& operator-=(value_type scalar) {
    std::for_each(begin(), end(), [&](reference e) { e -= scalar; });
    return *this;
  }

  // Replaces each and every element in the matrix by the result of
  // multiplication of that element and a `scalar`.
  inline self_type& operator*=(value_type scalar) {
    if (std::is_floating_point<T>::value) {
      internal::insight_scal(size(), scalar, begin());
    } else {
      std::for_each(begin(), end(), [&](reference e) { e *= scalar; });
    }
    return *this;
  }

  // Replaces each and every element in the matrix by the result of
  // dividing that element by a constant `scalar`.
  inline self_type& operator/=(value_type scalar) {
    if (std::is_floating_point<T>::value) {
      internal::insight_scal(size(), value_type(1.0) / scalar, begin());
    } else {
      std::for_each(begin(), end(), [&](reference e) { e /= scalar; });
    }
    return *this;
  }

  // matrix-matrix arithmetic.

  // Replaces each and every element in `this` matrix by the result of
  // adding that element with the corresponfing element in the `other`
  // matrix.
  inline self_type& operator+=(const matrix& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately by multiplying `this` by `2` for example?
    CHECK_EQ(num_rows(), other.num_rows());
    CHECK_EQ(num_cols(), other.num_cols());
    if (std::is_floating_point<T>::value) {
      internal::insight_add(size(), other.begin(), begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e += *it++; });
    }
    return *this;
  }

  // Replaces each and every element in `this` matrix by the result of
  // substracting the corresponding element in the `other` matrix from that
  // element.
  inline self_type& operator-=(const matrix& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately by setting all elements of `this` to `zero`?
    CHECK_EQ(num_rows(), other.num_rows());
    CHECK_EQ(num_cols(), other.num_cols());
    if (std::is_floating_point<T>::value) {
      internal::insight_sub(size(), begin(), other.begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e -= *it++; });
    }
    return *this;
  }

  // Replaces each and every element in `this` matrix by the result of
  // multiplying that element and the corresponding element in the
  // `other` matrix.
  inline self_type& operator*=(const matrix& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately?
    CHECK_EQ(num_rows(), other.num_rows());
    CHECK_EQ(num_cols(), other.num_cols());
    if (std::is_floating_point<T>::value) {
      internal::insight_mul(size(), other.begin(), begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e *= *it++; });
    }
    return *this;
  }

  // Replaces each and every element in `this` matrix by the result of
  // dividing that element by the corresponding element in the `other`
  // matrix.
  inline self_type& operator/=(const matrix& other) {
    // TODO(Linh): What happens if `this == &other`? Do we need to treat
    // this case separately by setting all elements of `this` to `one`?
    CHECK_EQ(num_rows(), other.num_rows());
    CHECK_EQ(num_cols(), other.num_cols());
    if (std::is_floating_point<T>::value) {
      internal::insight_div(size(), begin(), other.begin(), begin());
    } else {
      auto it = other.begin();
      std::for_each(begin(), end(), [&](reference e) { e /= *it++; });
    }
    return *this;
  }

  // Matrix expresison arithmetic.

  template<typename E>
  inline self_type& operator+=(const matrix_expression<E>& expr) {
    CHECK_EQ(num_rows(), expr.self().num_rows());
    CHECK_EQ(num_cols(), expr.self().num_cols());
    evaluator<E>::add(expr.self(), buffer::start);
    return *this;
  }

  template<typename E>
  inline self_type& operator-=(const matrix_expression<E>& expr) {
    CHECK_EQ(num_rows(), expr.self().num_rows());
    CHECK_EQ(num_cols(), expr.self().num_cols());
    evaluator<E>::sub(expr.self(), buffer::start);
    return *this;
  }

  template<typename E>
  inline self_type& operator*=(const matrix_expression<E>& expr) {
    CHECK_EQ(num_rows(), expr.self().num_rows());
    CHECK_EQ(num_cols(), expr.self().num_cols());
    evaluator<E>::mul(expr.self(), buffer::start);
    return *this;
  }

  template<typename E>
  inline self_type& operator/=(const matrix_expression<E>& expr) {
    CHECK_EQ(num_rows(), expr.self().num_rows());
    CHECK_EQ(num_cols(), expr.self().num_cols());
    evaluator<E>::div(expr.self(), buffer::start);
    return *this;
  }

 private:
  size_type num_rows_;
  size_type num_cols_;

  void destroy_elements() {
    if (std::is_trivially_destructible<T>::value) { return; }
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

// According to the Effective C++ swap idiom (item 25), we also need
// to provide a free swap function.
template<typename T, typename Allocator>
void swap(matrix<T, Allocator>& m1, matrix<T, Allocator>& m2) noexcept {
  m1.swap(m2);
}

}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_MATRIX_H_
