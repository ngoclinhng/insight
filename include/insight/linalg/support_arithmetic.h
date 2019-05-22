// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_SUPPORT_ARITHMETIC_H_
#define INCLUDE_INSIGHT_LINALG_SUPPORT_ARITHMETIC_H_

#include <type_traits>

#include "insight/linalg/fmatrix_base.h"
#include "insight/linalg/imatrix_base.h"

namespace insight {

struct not_support_arithmetic {};

template<typename Scalar, typename Matrix>
struct support_arithmetic {
  using type = typename std::conditional<
    std::is_floating_point<Scalar>::value,
    fmatrix_base<Matrix, Scalar>,
    typename std::conditional<std::is_integral<Scalar>::value,
                              imatrix_base<Matrix, Scalar>,
                              not_support_arithmetic>::type
    >::type;
};


}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_SUPPORT_ARITHMETIC_H_
