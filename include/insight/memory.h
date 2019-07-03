// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_MEMORY_H_
#define INCLUDE_INSIGHT_MEMORY_H_

#include "insight/internal/port.h"

#if defined(INSIGHT_USE_TBB_SCALABLE_MALLOC)
#include <tbb/scalable_allocator.h>
#else
#include <cstddef>
#include <utility>
#include "glog/logging.h"
#endif

#if defined(INSIGHT_USE_MKL_MALLOC)
#include <mkl.h>
#elif defined(INSIGHT_USE_POSIX_MEMALIGN)
#include <stdlib.h>
#elif defined(_MSC_VER)
#include <malloc.h>
#else
#include <cstdlib>
#endif

#include "insight/internal/type_traits.h"

namespace insight {

#if defined(INSIGHT_USE_TBB_SCALABLE_MALLOC)

template<typename T>
using allocator = tbb::scalable_allocator<T>;

#else

// custom allocator
template<typename T>
class allocator {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template<typename U> struct rebind { using other = allocator<U>; };

  allocator()  noexcept {}
  allocator(const allocator&)  noexcept { }
  template<typename U> allocator(const allocator<U>&)  noexcept {}

  pointer address(reference x) const {return &x;}
  const_pointer address(const_reference x) const {return &x;}

  // Allocate space for n objects.
  pointer allocate(size_type n, const void* = 0) {
    if (n == 0) { return NULL; }

    DLOG(INFO) << "allocating memory";

    const size_type n_bytes = n * sizeof(value_type);
    pointer p = NULL;
#if defined(INSIGHT_USE_MKL_MALLOC)
    p = reinterpret_cast<pointer>(mkl_malloc(n_bytes, 32));
#elif defined(INSIGHT_USE_POSIX_MEMALIGN)
    void* ptr = NULL;
    const size_type alignment = (n_bytes >= 1024) ? 32 : 16;
    int status = posix_memalign(&ptr, ((alignment >= sizeof(void*)) ?
                                       alignment : sizeof(void*) ), n_bytes);
    p = (status == 0) ? reinterpret_cast<pointer>(ptr) : NULL;
#elif defined(_MSC_VER)
    const size_type alignment = (n_bytes >= 1024) ? 32 : 16;
    p = reinterpret_cast<pointer>(_aligned_malloc(n_bytes, alignment));
#else
    p = reinterpret_cast<pointer>(std::malloc(n_bytes));
#endif
    if (!p) {
      LOG(FATAL) << "allocator: either requested size was too large or "
                 << "not enough available heap memory";
    }
    return p;
  }

  // Free previously allocated block of memory
  void deallocate(pointer p, size_type) {
    DLOG(INFO) << "deallocating memory";
#if defined(INSIGHT_USE_MKL_MALLOC)
    mkl_free(reinterpret_cast<void*>(p));
#elif defined(INSIGHT_USE_POSIX_MEMALIGN)
    free(reinterpret_cast<void*>(p));
#elif defined(_MSC_VER)
    _aligned_free(reinterpret_cast<void*>(p));
#else
    std::free(reinterpret_cast<void*>(p));
#endif
  }

  //! Largest value for which method allocate might succeed.
  size_type max_size() const noexcept {
    size_type absolutemax = static_cast<size_type>(-1) / sizeof (value_type);
    return (absolutemax > 0 ? absolutemax : 1);
  }

  template<typename U, typename... Args>
  void construct(U *p, Args&&... args) {
    ::new(reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
  }

  void destroy(pointer p) {
    p->~value_type();
  }
};  // allocator

template<>
class allocator<void> {
 public:
  using pointer = void*;
  using const_pointer = const void*;
  using value_type = void;

  template<class U> struct rebind { using  other = allocator<U>; };
};

template<typename T, typename U>
inline bool operator==(const allocator<T>&, const allocator<U>&) {
  return true;
}

template<typename T, typename U>
inline bool operator!=(const allocator<T>&, const allocator<U>&) {  // NOLINT
  return false;
}

#endif  // custom alllocator.

// Helper for conatiner swap. See [1] for reference
//
// [1] - https://en.cppreference.com/w/cpp/named_req/AllocatorAwareContainer

template<typename Alloc>
inline
void swap_allocator(Alloc& a1, Alloc& a2)  // NOLINT
    INSIGHT_NOEXCEPT_IF(internal::is_nothrow_swappable<Alloc>::value) {
  // allocator is replaced iff progagate_on_container_swap is true.
  swap_allocator(a1, a2, std::allocator_traits<Alloc>::progagate_on_container_swap::value);  // NOLINT
}

// The old allocator is replaced by the one in other container.
template<typename Alloc>
void swap_allocator(Alloc& a1, Alloc& a2, std::true_type)  // NOLINT
    INSIGHT_NOEXCEPT_IF(internal::is_nothrow_swappable<Alloc>::value) {
  using std::swap;
  swap(a1, a2);
}

// No swap happens. The old allocator is kept.
template<typename Alloc>
void swap_allocator(Alloc&, Alloc&, std::false_type) INSIGHT_NOEXCEPT {
};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_MEMORY_H_
