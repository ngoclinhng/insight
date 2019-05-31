// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_FUNCTIONAL_H_
#define INCLUDE_INSIGHT_LINALG_FUNCTIONAL_H_

#include <cmath>

namespace insight {
namespace unary_functor {

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
}  // namespace unary_functor
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_FUNCTIONAL_H_
