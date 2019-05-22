// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_STORAGE_H_
#define INCLUDE_INSIGHT_INTERNAL_STORAGE_H_

#include <utility>
#include <memory>

#include "insight/memory.h"

namespace insight {
namespace internal {

template<typename T, typename Allocator = insight_allocator<T> >
struct storage {
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using size_type = typename std::allocator_traits<Allocator>::size_type;

  // The allocator
  Allocator alloc;

  // start of allocation
  pointer start;

  // end of elements, start of space allocated for posible expansion.
  pointer  end;

  // End of allocated space.
  pointer last;

  storage(size_type n, const Allocator& a) : alloc(a),
                                             start(alloc.allocate(n)),
                                             end(start + n),
                                             last(start + n) {}

  ~storage() { alloc.deallocate(start, last - start); }

  // storage is noncopyable.
  storage() = default;
  storage(const storage&) = delete;
  storage& operator=(const storage&) = delete;

  // Increase the capacity of the storage to a value that is greater than
  // or equal `new_cap`. If `new_cap` is greater than the current capacity
  // (last - start), new storage is allocated, otherwise this method does
  // nothing.
  void reserve(size_type new_cap) {
    size_type current_cap = last - start;
    if (new_cap <= current_cap) { return; }

    // Allocate new storage.
    pointer new_start = alloc.allocate(new_cap);

    // Copy data over.
    size_type count = end - start;
    std::uninitialized_copy_n(start, count, new_start);

    // Destroy elements in the old storage.
    for (pointer p = start; p != end; ++p) { p->~T(); }

    // Free old storage.
    alloc.deallocate(start, last - start);

    // Reset storage.
    start = new_start;
    end = new_start + count;
    last = new_start + new_cap;
  }

  void swap(storage<T, Allocator>& other) noexcept {
    using std::swap;
    swap(alloc, other.alloc);
    swap(start, other.start);
    swap(end, other.end);
    swap(last, other.last);
  }
};

// We should also provide a non-member swap according to the Effective C++
// swap idiom.
template<typename T, typename Allocator>
void swap(storage<T, Allocator>& s1, storage<T, Allocator>& s2) noexcept {
  s1.swap(s2);
}

}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_STORAGE_H_
