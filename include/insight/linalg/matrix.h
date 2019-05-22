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
#include "insight/linalg/support_arithmetic.h"
#include "insight/linalg/operator_times.h"

#include "glog/logging.h"

namespace insight {

template<typename T, typename Allocator = insight_allocator<T> >
class matrix:
      public support_arithmetic<T, matrix<T, Allocator> >::type,
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

  // Return the number of rows in `this` matrix.
  size_type num_rows() const { return num_rows_; }

  // Returns the number of columns in `this` matrix.
  size_type num_cols() const { return num_cols_; }

  // Returns the `(num_rows, num_cols)` pair that represents the
  // shape/dimensions of `this` matrix.
  shape_type shape() const {
    return shape_type(num_rows_, num_cols_);
  }

  // Returns the number of elements in `this` matrix.
  size_type size() const { return (buffer::end - buffer::start); }

  // Returns true if this matrix is empty.
  bool empty() const { return size() == 0;}

  // Returns the number of elements that the matrix has currently
  // allocated space for.
  size_type capacity() const { return (buffer::last - buffer::start); }

  //
  // Returns the underlying allocator.
  allocator_type get_allocator() const { return buffer::alloc; }

  // Accesses the element at index i in the underlying buffer without
  // bounce-checking. A reference is returned.
  reference operator[](const size_type i) { return buffer::start[i]; }

  // Accesses the element at index i in the underlying buffer without
  // bounce-checking. A const reference is returned.
  const_reference operator[](const size_type i) const {
    return buffer::start[i];
  }

  // Accesses the element at row `i` and column `j` in the matrix without
  // bounce-checking. A reference is returned.
  reference operator()(const size_type i, const size_type j) {
    return buffer::start[i * num_cols_ + j];
  }

  // Accesses the element at row `i` and column `j` in the matrix without
  // bounce-checking. A const reference is returned.
  const_reference operator()(const size_type i, const size_type j) const {
    return buffer::start[i * num_cols_ + j];
  }

  // Element-wise iterator

  iterator begin() { return buffer::start; }
  const_iterator begin() const { return buffer::start; }
  const_iterator cbegin() const { return buffer::start; }

  iterator end() { return buffer::end; }
  const_iterator end() const { return buffer::end; }
  const_iterator cend() const { return buffer::end; }

  // Swap two matrices.
  void swap(self_type& other) noexcept {
    using std::swap;
    swap(static_cast<buffer&>(*this), static_cast<buffer&>(other));
    swap(num_rows_, other.num_rows_);
    swap(num_cols_, other.num_cols_);
  }


  // row view
  // ==================================================================

  class row_view: public support_arithmetic<T, row_view>::type {
    using base_type = matrix<T, Allocator>;

   public:
    using value_type = typename base_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::const_pointer;

    using shape_type = typename base_type::shape_type;

    row_view() = default;
    ~row_view() = default;

    row_view(base_type* base, size_type index)
        : base_(base),
          row_start_(base->begin() + index * base->num_cols()),
          row_end_(base->begin() + (index + 1) * base->num_cols()) {
      CHECK_LT(index, base->num_rows()) << "row_view: invalid row index";
    }

    size_type num_rows() const { return 1; }

    size_type num_cols() const { return base_->num_cols(); }

    shape_type shape() const {
      return shape_type(1, num_cols());
    }

    size_type size() const { return num_cols(); }

    bool empty() const { return size() == 0; }

    reference operator[](const size_type i) {
      return row_start_[i];
    }

    const_reference operator[](const size_type i) const {
      return row_start_[i];
    }

    // iterator.

    using iterator = pointer;
    using const_iterator = const_pointer;

    iterator begin() { return row_start_; }
    const_iterator begin() const { return row_start_; }
    const_iterator cbegin() const { return row_start_; }

    iterator end() { return row_end_; }
    const_iterator end() const { return row_end_; }
    const_iterator cend() const { return row_end_; }

    bool aliased_of(const row_view& other) const {
      return (this == &other) || (row_start_ == other.row_start_);
    }
    bool aliased_of(const base_type& base) const {
      return (base_ == &base);
    }

   private:
    base_type* base_;
    pointer row_start_;
    pointer row_end_;
  };

  // Accesses the row at index `row_index`.
  row_view row_at(size_type row_index) {
    return row_view(this, row_index);
  }

  // Returns true if `this` and `other` are indeed the same matrix.
  bool aliased_of(const matrix& other) const { return (this == &other); }
  bool aliased_of(const row_view& view) const {
    return view.aliased_of(*this);
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
