// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/function_sample.h"

namespace insight {
namespace internal {

function_sample::function_sample()
    : trial_step(0.0),
      next_iterate(),
      value_of_objective_function(0.0),
      gradient_of_objective_function(),
      gradient_of_phi_function(0.0) {}

function_sample::function_sample(double trial_step,
                                 double value_of_objective_function)
    : trial_step(trial_step),
      next_iterate(),
      value_of_objective_function(value_of_objective_function),
      gradient_of_objective_function(),
      gradient_of_phi_function(0.0) {}

function_sample::function_sample(double trial_step,
                                 double value_of_objective_function,
                                 double gradient_of_phi_function)
    : trial_step(trial_step),
      next_iterate(),
      value_of_objective_function(value_of_objective_function),
      gradient_of_objective_function(),
      gradient_of_phi_function(gradient_of_phi_function) {}

}  // namespace internal
}  // namespace insight
