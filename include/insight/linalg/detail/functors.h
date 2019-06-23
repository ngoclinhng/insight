// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_FUNCTORS_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_FUNCTORS_H_

#include <cmath>

namespace insight {
namespace linalg_detail {

template<typename T>
struct sqrt {
  inline T operator()(T value) const { return std::sqrt(value); }
};

template<typename T>
struct exp {
  inline T operator()(T value) const { return std::exp(value); }
};

template<typename T>
struct log {
  inline T operator()(T value) const { return std::log(value); }
};

}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_FUNCTORS_H_
