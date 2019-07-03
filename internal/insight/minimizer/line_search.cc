// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/minimizer/objective_functions.h"
#include "insight/line_search.h"

namespace insight {
namespace internal {

phi_function::phi_function(first_order_function* objective_function)
    : objective_function_(objective_function),
      current_iterate_(objective_function->parameter_count()),
      search_direction_(objective_function->parameter_count()),
      scaled_search_direction_(objective_function->parameter_count()) {}

void phi_function::init(const vector<double>& current_iterate,
                        const vector<double>& search_direction) {
  current_iterate_ = current_iterate;
  search_direction_ = search_direction;
}

void phi_function::evaluate(double trial_step,
                            bool evaluate_gradient,
                            function_sample* result) {
  result->trial_step = trial_step;
  scaled_search_direction_ = result->trial_step * search_direction_;
  result->next_iterate = current_iterate_ + scaled_search_direction_;

  double* gradient = NULL;
  if (evaluate_gradient) {
    result->gradient_of_objective_function
        = vector<double>(objective_function_->parameter_count());
    gradient = result->gradient_of_objective_function.data();
  }

  // Compute the value and gradient of the objective function at the
  // next_iterate.
  objective_function_->evaluate(result->next_iterate.data(),
                                &(result->value_of_objective_function),
                                gradient);

  // Compute the gradient of phi function at the trial step length
  // trial_step
  result->gradient_of_phi_function =
      search_direction_.dot(result->gradient_of_objective_function);
}

}  // namespace internal
}  // namespace insight
