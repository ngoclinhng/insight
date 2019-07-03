// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_MINIMIZER_OBJECTIVE_FUNCTIONS_H_
#define INCLUDE_INSIGHT_MINIMIZER_OBJECTIVE_FUNCTIONS_H_

#include "insight/internal/port.h"

namespace insight {

// A first_order_function represents a real-valued, differentiable function:
//
//  f : R^n                      ---> R
//      x = (x_1, x_2, ..., x_n) ---> f(x1, x2, ..., x_n)
class INSIGHT_EXPORT first_order_function {
 public:
  virtual ~first_order_function() {}

  // Returns the number of variables
  virtual int parameter_count() const = 0;

  // Evaluate the value and (optionally) the gradient of the function
  // at the given point.
  virtual void evaluate(const double* x, double* value, double* gradient);
};
}  // namespace insight
#endif  // INCLUDE_INSIGHT_MINIMIZER_OBJECTIVE_FUNCTIONS_H_
