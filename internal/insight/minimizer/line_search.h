// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INTERNAL_INSIGHT_LINE_SEARCH_H_
#define INTERNAL_INSIGHT_LINE_SEARCH_H_

#include "insight/linalg/vector.h"
#include "insight/function_sample.h"

namespace insight {

// The type of the objective function used by line search.
class first_order_function;

namespace internal {

// Given the objective function f of type first_order_function, the
// current_iterate x_k and the search direction d_k, instances of
// phi_function represents the following univariate function:
//
//  phi(alpha) = f(x_k + alpha * d_k), alpha > 0.
class phi_function {
 public:
  explicit phi_function(first_order_function* objective_function);

  void init(const vector<double>& current_irerate,
            const vector<double>& search_direction);  // NOLINT

  // Evaluate the value and (optionally) gradient of the objective_function
  // at the next_iterate = current_iterate + trial_step * search_direction.
  void evaluate(double trial_step, bool evaluate_gradient,
                function_sample* result);

  // The infinity norm of the search_direction_ vector.
  double search_direction_infinity_norm() const;

 private:
  first_order_function* objective_function_;
  vector<double> current_iterate_;
  vector<double> search_direction_;

  // scaled_search_direction = trial_step * search_direction_
  vector<double> scaled_search_direction_;  // NOLINT
};
}  // namespace internal
}  // namespace insight
#endif  // INTERNAL_INSIGHT_LINE_SEARCH_H_
