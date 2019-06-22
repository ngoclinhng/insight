// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_DENSE_BASE_H_
#define INCLUDE_INSIGHT_INTERNAL_DENSE_BASE_H_

#include <utility>
#include <memory>
#include <type_traits>
#include <stdexcept>

#include "insight/internal/port.h"

namespace insight {
namespace internal {
template<typename T, typename Alloc>
class dense_base {
 public:
  using allocator_type = typename std::allocator_traits<Alloc>::template
                     rebind_alloc<T>;
  using alloc_traits = std::allocator_traits<allocator_type>;
  using size_type = typename alloc_traits::size_type;

 protected:
  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using difference_type = typename alloc_traits::difference_type;
  using pointer = typename alloc_traits::pointer;
  using const_pointer = typename alloc_traits::const_pointer;
  using iterator = pointer;
  using const_iterator = const_pointer;

  pointer begin_;    // start of the allocated space
  pointer end_;      // end of elements
  pointer end_cap_;  // end of the allocated space

  allocator_type alloc_;

  dense_base() INSIGHT_NOEXCEPT_IF(
      std::is_nothrow_default_constructible<allocator_type>::value);

  dense_base(const allocator_type& a) INSIGHT_NOEXCEPT_IF(  // NOLINT
      std::is_nothrow_copy_constructible<allocator_type>::value);

  dense_base(allocator_type&& a) INSIGHT_NOEXCEPT;  // NOLINT

  ~dense_base();

  // Clears the contents of the container.
  void clear() INSIGHT_NOEXCEPT {destruct_at_end_(begin_); }

  // Returns the number of elements in the container.
  size_type size() const INSIGHT_NOEXCEPT {
    return static_cast<size_type>(end_ - begin_);
  }

  // Returns the number of elements that the container has currently
  // allocated space for.
  size_type capacity() const INSIGHT_NOEXCEPT {
    return static_cast<size_type>(end_cap_ - begin_);
  }

  // Destroys all the elements pointed to by pointers in the range
  // [new_end, end_) in reverse order (meaning going from element just
  // before end_ downto the element pointed to by new_end),
  // and reset end_ to new_end.
  void destruct_at_end_(pointer new_end) INSIGHT_NOEXCEPT;

  // Use by copy-assignment to replace allocator.
  void copy_assign_alloc_(const dense_base& b) {
    copy_assign_alloc_(b, std::integral_constant<bool, alloc_traits::propagate_on_container_copy_assignment::value>());  // NOLINT
  }

  // Use by move-assignment to replace allocator.
  void move_assign_alloc_(dense_base& b) INSIGHT_NOEXCEPT_IF(  // NOLINT
      !alloc_traits::propagate_on_container_move_assignment::value ||
      std::is_nothrow_move_assignable<allocator_type>::value) {
    move_assign_alloc_(b, std::integral_constant<bool, alloc_traits::propagate_on_container_move_assignment::value>());  // NOLINT
  }

  NO_RETURN void throw_length_error(const char* what) const {
    throw std::length_error(what);
  }

  NO_RETURN void throw_out_of_range(const char* what) const {
    throw std::out_of_range(what);
  }

 private:
  // Destroys all elements in the range [new_end, end_) for value_type that
  // is trivially destructible.
  void destruct_at_end_(pointer new_end, std::true_type) INSIGHT_NOEXCEPT {
    end_ = new_end;
  }

  // Destroys all elements in the range [new_end, end_) for value_type that
  // is NOT trivially destructible.
  void destruct_at_end_(pointer new_end, std::false_type) INSIGHT_NOEXCEPT {
    pointer soon_to_be_end = end_;
    while (new_end != soon_to_be_end) {
      alloc_traits::destroy(alloc_, --soon_to_be_end);
    }
    end_ = new_end;
  }

  // Copies alloc iff alloc_traits::propagate_on_container_copy_assignment
  // ::value is true.
  void copy_assign_alloc_(const dense_base& b, std::true_type) {
    if (alloc_ != b.alloc_) {
      clear();
      alloc_traits::deallocate(alloc_, begin_, capacity());
    }
    alloc_ = b.alloc_;
  }

  // If alloc_traits::propagate_on_container_copy_assignment::value is false,
  // no copy happens!
  void copy_assign_alloc_(const dense_base&, std::false_type) {}

  // Moves alloc iff alloc_traits::propagate_on_container_move_assignment
  // ::value is true.
  void move_assign_alloc_(dense_base& b, std::true_type)  // NOLINT
      INSIGHT_NOEXCEPT_IF(
          std::is_nothrow_move_assignable<allocator_type>::value) {
    alloc_ = std::move(b.alloc_);
  }

  // If alloc_traits::propagate_on_container_move_assignment::value is false,
  // simple does nothing.
  void move_assign_alloc_(dense_base&, std::false_type) INSIGHT_NOEXCEPT {}
};

template<typename T, typename Alloc>
inline
dense_base<T, Alloc>::dense_base()
    INSIGHT_NOEXCEPT_IF(std::is_nothrow_default_constructible<allocator_type>::value)  // NOLINT
    : begin_(nullptr),
      end_(nullptr),
      end_cap_(nullptr),
      alloc_() {
}

template<typename T, typename Alloc>
inline
dense_base<T, Alloc>::dense_base(const allocator_type& a)
    INSIGHT_NOEXCEPT_IF(std::is_nothrow_copy_constructible<allocator_type>::value)  // NOLINT
    : begin_(nullptr),
      end_(nullptr),
      end_cap_(nullptr),
      alloc_(a) {
}

template<typename T, typename Alloc>
inline
dense_base<T, Alloc>::dense_base(allocator_type&& a) INSIGHT_NOEXCEPT
    : begin_(nullptr),
      end_(nullptr),
      end_cap_(nullptr),
      alloc_(std::move(a)) {
}

template<typename T, typename Alloc>
dense_base<T, Alloc>::~dense_base() {
  if (begin_ != nullptr) {
    clear();
    alloc_traits::deallocate(alloc_, begin_, capacity());
  }
}

template<typename T, typename Alloc>
inline
void
dense_base<T, Alloc>::destruct_at_end_(pointer new_end) INSIGHT_NOEXCEPT {
  destruct_at_end_(new_end, std::integral_constant<bool, std::is_trivially_destructible<value_type>::value>());  // NOLINT
}

}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_DENSE_BASE_H_
