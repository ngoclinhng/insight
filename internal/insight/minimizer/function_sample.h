// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INTERNAL_INSIGHT_FUNCTION_SAMPLE_H_
#define INTERNAL_INSIGHT_FUNCTION_SAMPLE_H_

#include "insight/linalg/vector.h"

namespace insight {
namespace internal {

// function_sample is used by the line search routines to store and
// communicate the value and gradient of the function being minimized
struct function_sample {
  function_sample();
  function_sample(double trial_step,
                  double value_of_objective_function);
  function_sample(double trial_step,
                  double value_of_objective_function,
                  double gradient_of_phi_function);

  // The trial step length along the search direction.
  double trial_step;

  // Next iterate from the current iterate:
  //
  //  next_iterate = current_iterate + trial_step * search_direction.
  vector<double> next_iterate;  // NOLINT

  // The value of the objective function at the next_iterate.
  //
  //  value_of_objective-function = f(next_iterate).
  double value_of_objective_function;

  // The gradient of the objective function at the next_iterate.
  //
  //  gradient_of_objective_function = f'(next_iterate)
  vector<double> gradient_of_objective_function;  // NOLINT

  // The gradient of the phi function at trial_step.
  //
  // By definition:
  //
  //  phi(alpha) = f(x_k + alpha * d_k), alpha > 0 where x_k, d_k are
  //  the current iterate and the search direction, respectively.
  //
  // Hence:
  //
  //  phi'(alpha) = grad_f(x_k + alpha * d_k).dot(d_k).
  double gradient_of_phi_function;
};
}  // namespace internal
}  // namespace insight
#endif  // INTERNAL_INSIGHT_FUNCTION_SAMPLE_H_
