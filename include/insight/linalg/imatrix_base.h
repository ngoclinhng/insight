// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_IMATRIX_BASE_H_
#define INCLUDE_INSIGHT_LINALG_IMATRIX_BASE_H_

namespace insight {

template<typename Derived, typename VT>
struct imatrix_base {
  Derived& self() { return static_cast<Derived&>(*this); }
};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_IMATRIX_BASE_H_
