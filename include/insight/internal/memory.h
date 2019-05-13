// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_MEMORY_H_
#define INCLUDE_INSIGHT_INTERNAL_MEMORY_H_

#include <cstdlib>

#include "insight/internal/port.h"

#if defined(INSIGHT_USE_TBB_SCALABLE_MALLOC)
#include <tbb/scalable_allocator.h>
#elif defined(INSIGHT_USE_MKL_MALLOC)
#include <mkl.h>
#elif defined(INSIGHT_USE_POSIX_MEMALIGN)
#include <stdlib.h>
#elif defined(_MSC_VER)
#include <malloc.h>
#endif

#include "glog/logging.h"

namespace insight {
namespace internal {

template<typename T>
inline T* insight_malloc(const std::size_t n_elements) {
  if (n_elements == 0) { return NULL; }

  const std::size_t n_bytes = sizeof(T) * n_elements;
  T* retptr = NULL;

#if defined(INSIGHT_USE_TBB_SCALABLE_MALLOC)
  retptr = reinterpret_cast<T*>(scalable_malloc(n_bytes));
#elif defined(INSIGHT_USE_MKL_MALLOC)
  retptr = reinterpret_cast<T*>(mkl_malloc(n_bytes, 32));
#elif defined(INSIGHT_USE_POSIX_MEMALIGN)
  void* ptr = NULL;
  const std::size_t alignment = (n_bytes >= 1024) ? 32 : 16;
  int status = posix_memalign(&ptr, ((alignment >= sizeof(void*)) ?
                                     alignment : sizeof(void*) ), n_bytes);
  retptr = (status == 0) ? reinterpret_cast<T*>(ptr) : NULL;
#elif defined(_MSC_VER)
  const std::size_t alignment = (n_bytes >= 1024) ? 32 : 16;
  retptr = reinterpret_cast<T*>(_aligned_malloc(n_bytes, alignment));
#else
  retptr = reinterpret_cast<T*>(std::malloc(n_bytes));
#endif

  CHECK_NOTNULL(retptr);
  return retptr;
}

template<typename T>
inline void insight_free(T* ptr) {
#if defined(INSIGHT_USE_TBB_SCALABLE_MALLOC)
  scalable_free(reinterpret_cast<void*>(ptr));
#elif defined(INSIGHT_USE_MKL_MALLOC)
  mkl_free(reinterpret_cast<void*>(ptr));
#elif defined(INSIGHT_USE_POSIX_MEMALIGN)
  free(reinterpret_cast<void*>(ptr));
#elif defined(_MSC_VER)
  _aligned_free(reinterpret_cast<void*>(ptr));
#else
  std::free(reinterpret_cast<void*>(ptr));
#endif
}

}  // namespace internal
}  // namespace insight

#endif  // INCLUDE_INSIGHT_INTERNAL_MEMORY_H_
