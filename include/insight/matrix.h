// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_MATRIX_H_
#define INCLUDE_INSIGHT_MATRIX_H_

#include <cstddef>
#include <utility>
#include <initializer_list>

namespace insight {

// Dense, row-major matrix of values of type T where T is floating point.
template<typename T>
class matrix {
 public:
  using size_type = std::size_t;
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;

  // Constructs an empty dense matrix.
  matrix();

  // Constructs a dense matrix with the specified dimensions. All elements
  // are initialized to `value`.
  //
  // Note that: if either one of `num_rows` or `num_cols` or both equal
  // to 0, than this will construct an empty matrix.
  matrix(const size_type num_rows, const size_type num_cols,
         const T& value = T());

  // Constructs a `1` by `n` dense matrix where `n` is the number of elements
  // in the initializer list `list`.
  matrix(const std::initializer_list<T>& list);

  matrix<T>& operator=(const std::initializer_list<T>& list);

  // Constructs a `m` by `n` matrix where `m` is the number of sublists in
  // the specified `list`, and `n` is the size of each sublist.
  matrix(const std::initializer_list< std::initializer_list<T> >& list);

  matrix<T>& operator=(const std::initializer_list< std::initializer_list<T> >& list);  // NOLINT

  // Copy constructor.
  matrix(const matrix<T>& src);


  // Swap two dense matrices.
  template<typename U>
  friend void swap_matrix(matrix<U>& m1, matrix<U>& m2) noexcept;  // NOLINT

  // Assignment operator.
  matrix<T>& operator=(const matrix<T>& rhs);

  // Move contructor
  matrix(matrix<T>&& src) noexcept;

  // Move assignment operator.
  matrix<T>& operator=(matrix<T>&& rhs) noexcept;

  // Destructor.
  ~matrix();

  // Returns the number of rows of the matrix.
  size_type num_rows() const { return num_rows_; }

  // Returns the number of columns of the matrix.
  size_type num_cols() const { return num_cols_; }

  // Returns the number of elements in the matrix.
  size_type size() const { return num_elem_; }

  // Returns the shape of the matrix.
  std::pair<size_type, size_type> shape() const {
    return std::pair<size_type, size_type>(num_rows_, num_cols_);
  }

  // Accesses the element at index i in the underlying buffer without
  // bounce-checking. A reference is returned.
  reference operator[](const size_type i) { return buffer_[i]; }

  // Accesses the element at index i in the underlying buffer without
  // bounce-checking. A const reference is returned.
  const_reference operator[](const size_type i) const { return buffer_[i]; }

  // Accesses the element at row `i` and column `j` in the matrix without
  // bounce-checking. A reference is returned.
  reference operator()(const size_type i, const size_type j) {
    return buffer_[i * num_cols_ + j];
  }

  // Accesses the element at row `i` and column `j` in the matrix without
  // bounce-checking. A const reference is returned.
  const_reference operator()(const size_type i, const size_type j) const {
    return buffer_[i * num_cols_ + j];
  }

  // Element-wise iterator
  // ----------------------------------------------------------------------

  using iterator = T*;
  using const_iterator = const T*;

  iterator begin() { return buffer_; }
  const_iterator begin() const { return buffer_; }
  const_iterator cbegin() const { return buffer_; }

  iterator end() { return buffer_ + num_elem_; }
  const_iterator end() const { return buffer_ + num_elem_; }
  const_iterator cend() const { return buffer_ + num_elem_; }

 private:
  T* buffer_;
  size_type num_elem_;
  size_type num_rows_;
  size_type num_cols_;
};

}  // namespace insight

#endif  // INCLUDE_INSIGHT_MATRIX_H_
