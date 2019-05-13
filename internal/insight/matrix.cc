// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <algorithm>

#include "insight/matrix.h"
#include "insight/memory.h"
#include "glog/logging.h"

namespace insight {

template<typename T>
matrix<T>::matrix() : buffer_(nullptr),
                      num_elem_(0),
                      num_rows_(0),
                      num_cols_(0) {}

template<typename T>
matrix<T>::matrix(const size_type num_rows, const size_type num_cols,
                  const T& value)
    : buffer_(internal::insight_malloc<T>(num_rows * num_cols)),
      num_elem_(num_rows * num_cols),
      num_rows_(num_elem_ > 0 ? num_rows : 0),
      num_cols_(num_elem_ > 0 ? num_cols : 0) {
  std::fill(buffer_, buffer_ + num_elem_, value);
}

template<typename T>
matrix<T>::matrix(const std::initializer_list<T>& list)
    : buffer_(internal::insight_malloc<T>(list.size())),
      num_elem_(list.size()),
      num_rows_(num_elem_ > 0 ? 1 : 0),
      num_cols_(list.size()) {
  std::copy(list.begin(), list.end(), buffer_);
}

template<typename T>
matrix<T>& matrix<T>::operator=(const std::initializer_list<T>& list) {
  // We only allocate memory if the current storage of `this` is
  // smaller than  `list.size()` or the `list` being assigned to is empty.
  if ((list.size() == 0) || (num_elem_ < list.size())) {
    matrix<T> temp(list);
    swap_matrix(*this, temp);
    return *this;
  }

  // Otherwise, resue memory.
  std::copy(list.begin(), list.end(), buffer_);
  num_elem_ = list.size();
  num_rows_ = 1;
  num_cols_ = list.size();
  return *this;
}

template<typename T>
matrix<T>::matrix(const std::initializer_list< std::initializer_list<T> >& list)
    : buffer_(nullptr),
      num_elem_(0),
      num_rows_(0),
      num_cols_(0) {
  if (list.size() == 0) { return; }

  // Make sure all sub-lists are of the same size.
  auto sublist_size = list.begin()->size();
  CHECK_GT(sublist_size, 0);
  for (auto it = list.begin(); it != list.end(); ++it) {
    CHECK_EQ(it->size(), sublist_size);
  }

  buffer_ = internal::insight_malloc<T>(list.size() * sublist_size);
  T* buffer_start = buffer_;
  for (auto it = list.begin(); it != list.end(); ++it) {
    std::copy(it->begin(), it->end(), buffer_start);
    buffer_start += sublist_size;
  }

  num_elem_ = list.size() * sublist_size;
  num_rows_ = list.size();
  num_cols_ = sublist_size;
}

template<typename T>
matrix<T>& matrix<T>::operator=(const std::initializer_list< std::initializer_list<T> >& list) {  // NOLINT
  // If the specified `list` is empty, we simply delete the old content,
  // and contruct an empty matrix.
  if (list.size() == 0) {
    matrix<T> temp(list);
    swap_matrix(*this, temp);
    return *this;
  }

  // Make sure all sub-lists are of the same size.
  // TODO(Linh): Redundance code!
  auto sublist_size = list.begin()->size();
  CHECK_GT(sublist_size, 0);
  for (auto it = list.begin(); it != list.end(); ++it) {
    CHECK_EQ(it->size(), sublist_size);
  }

  // We only allocate new memory if the current storage of `this` is
  // smaller than `list.size() * sublist_size`.
  auto new_num_elem = list.size() * sublist_size;
  if (num_elem_ < new_num_elem) {
    matrix<T> temp(list);
    swap_matrix(*this, temp);
    return *this;
  }

  // Otherwise, resue memory.
  T* start_buffer = buffer_;
  for (auto it = list.begin(); it != list.end(); ++it) {
    std::copy(it->begin(), it->end(), start_buffer);
    start_buffer += sublist_size;
  }
  num_elem_ = new_num_elem;
  num_rows_ = list.size();
  num_cols_ = sublist_size;
  return *this;
}

template<typename T>
matrix<T>::matrix(const matrix<T>& src)
    : buffer_(internal::insight_malloc<T>(src.num_elem_)),
      num_elem_(src.num_elem_),
      num_rows_(src.num_rows_),
      num_cols_(src.num_cols_) {
  std::copy(src.buffer_, src.buffer_ + num_elem_, buffer_);
}

template<typename U>
void swap_matrix(matrix<U>& m1, matrix<U>& m2) noexcept {  // NOLINT
  std::swap(m1.buffer_, m2.buffer_);
  std::swap(m1.num_elem_, m2.num_elem_);
  std::swap(m1.num_rows_, m2.num_rows_);
  std::swap(m1.num_cols_, m2.num_cols_);
}

template<typename T>
matrix<T>& matrix<T>::operator=(const matrix<T>& rhs) {
  // Early exit for self-assignment.
  if (this == &rhs) { return *this; }

  // We only allocate new memory if the current storage of `this` is
  // smaller than that of rhs or the rhs is empty.
  if ((rhs.num_elem_ == 0) || (num_elem_ < rhs.num_elem_)) {
    matrix<T> temp(rhs);
    swap_matrix(*this, temp);
    return *this;
  }

  // Otherwise, we reuse the storage of `this`.
  std::copy(rhs.buffer_, rhs.buffer_ + rhs.num_elem_, buffer_);
  num_elem_ = rhs.num_elem_;
  num_rows_ = rhs.num_rows_;
  num_cols_ = rhs.num_cols_;
  return *this;
}

template<typename T>
matrix<T>::matrix(matrix<T>&& src) noexcept
    : buffer_(nullptr),
      num_elem_(0),
      num_rows_(0),
      num_cols_(0) {
  swap_matrix(*this, src);
}

template<typename T>
matrix<T>& matrix<T>::operator=(matrix<T>&& rhs) noexcept {
  if (this == &rhs) { return *this; }
  swap_matrix(*this, rhs);
  return *this;
}
template<typename T>
matrix<T>::~matrix() {
  internal::insight_free<T>(buffer_);
  buffer_ = nullptr;
}

// We explicitly instantiate matrix for float and double data type.
template class matrix<float>;
template class matrix<double>;

}  // namespace insight
