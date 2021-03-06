// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_MUL_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_MUL_H_

#include "insight/linalg/detail/special_expression_traits.h"


namespace insight {
namespace linalg_detail {

template<typename E> struct is_special_multiplicable
    : public std::false_type{};

namespace special_expression {

// placeholder.
template<typename E>
inline
void mul(const E&, typename E::value_type*) {
}

}  // namespace special_expression
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_SPECIAL_EXPRESSION_MUL_H_
