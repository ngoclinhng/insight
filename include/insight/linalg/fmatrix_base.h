// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_FMATRIX_BASE_H_
#define INCLUDE_INSIGHT_LINALG_FMATRIX_BASE_H_

#include <algorithm>

#include "insight/internal/math_functions.h"
#include "glog/logging.h"

namespace insight {

// Forword declaration
template<typename L, typename R> struct mul_expr;

// Base class for all floating-point matrix.
template<typename Derived, typename VT>
struct fmatrix_base {
  const Derived& self() const { return static_cast<const Derived&>(*this); }
  Derived& mutable_self() { return static_cast<Derived&>(*this); }

  inline Derived& operator+=(VT scalar) {
    std::for_each(mutable_self().begin(), mutable_self().end(),
                  [&](VT& e){ e += scalar; });
    return mutable_self();
  }

  inline Derived& operator-=(VT scalar) {
      std::for_each(mutable_self().begin(), mutable_self().end(),
                    [&](VT& e){ e -= scalar; });
    return mutable_self();
  }

  inline Derived& operator*=(VT scalar) {
      internal::insight_scal(mutable_self().size(), scalar,
                             mutable_self().begin());
    return mutable_self();
  }

  inline Derived& operator/=(VT scalar) {
      internal::insight_scal(mutable_self().size(), VT(1.0)/scalar,
                             mutable_self().begin());
    return mutable_self();
  }

  template<typename Other>
  inline Derived& operator+=(const fmatrix_base<Other, VT>& m) {
    CHECK_EQ(self().num_rows(), m.self().num_rows());
    CHECK_EQ(self().num_cols(), m.self().num_cols());

    if (self().aliased_of(m.self())) {
      operator*=(VT(2.0));
    } else {
      internal::insight_axpy(self().size(), VT(1.0), m.self().begin(),
                             mutable_self().begin());
    }
    return mutable_self();
  }

  template<typename Other>
  inline Derived& operator-=(const fmatrix_base<Other, VT>& m) {
    CHECK_EQ(self().num_rows(), m.self().num_rows());
    CHECK_EQ(self().num_cols(), m.self().num_cols());

    if (self().aliased_of(m.self())) {
      std::fill(mutable_self().begin(), mutable_self().end(), VT(0.0));
    } else {
      internal::insight_axpy(self().size(), VT(-1.0), m.self().begin(),
                             mutable_self().begin());
    }
    return mutable_self();
  }

  // Y += a * X.
  template<typename Other>
  inline
  Derived& operator+=(const mul_expr<fmatrix_base<Other, VT>, VT>& expr) {
    CHECK_EQ(self().num_rows(), expr.matrix_ref().num_rows());
    CHECK_EQ(self().num_cols(), expr.matrix_ref().num_cols());

    if (expr.matrix_ref().aliased_of(self())) {
      // X += a * X.
      operator*=(VT(1.0) + expr.scalar);
    } else {
      // Y += a * X.
      internal::insight_axpy(self().size(), expr.scalar,
                             expr.matrix_ref().begin(),
                             mutable_self().begin());
    }
    return mutable_self();
  }

  // Y -= a * X.
  template<typename Other>
  inline
  Derived& operator-=(const mul_expr<fmatrix_base<Other, VT>, VT>& expr) {
    CHECK_EQ(self().num_rows(), expr.matrix_ref().num_rows());
    CHECK_EQ(self().num_cols(), expr.matrix_ref().num_cols());

    if (expr.matrix_ref().aliased_of(self())) {
      // X -= a * X.
      operator*=(VT(1.0) - expr.scalar);
    } else {
      // Y -= a * X.
      internal::insight_axpy(self().size(), -expr.scalar,
                             expr.matrix_ref().begin(),
                             mutable_self().begin());
    }
    return mutable_self();
  }
};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_FMATRIX_BASE_H_
