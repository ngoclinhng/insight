// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_EVALUATOR_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_EVALUATOR_H_

#include <algorithm>

#include "insight/linalg/detail/special_expression_assign.h"
#include "insight/linalg/detail/special_expression_add.h"
#include "insight/linalg/detail/special_expression_sub.h"
#include "insight/linalg/detail/special_expression_mul.h"
#include "insight/linalg/detail/special_expression_div.h"

namespace insight {
namespace linalg_detail {

// Evaluate a generic expression.
template<typename E>
struct expression_evaluator {
  using value_type = typename E::value_type;
  const E& e;

  explicit expression_evaluator(const E& e) : e(e) {}

  // Evaluates the expression e, and copies the result to buffer.
  inline void assign(value_type* buffer) const {
    assign_(buffer, std::integral_constant<bool, is_special_assignable<E>::value>());  // NOLINT
  }

  // buffer += e: evaluates the expression e, and adds the result to the
  // buffer (element-wise).
  inline void add(value_type* buffer) const {
    add_(buffer, std::integral_constant<bool, is_special_addable<E>::value>());
  }

  // buffer -= e: evaluates the expression e, and subtracts the result from
  // the buffer (element-wise).
  inline void sub(value_type* buffer) const {
    sub_(buffer, std::integral_constant<bool, is_special_subtractable<E>::value>());  // NOLINT
  }

  // buffer *= e: evaluates the expression e, and multiplies the result with
  // buffer (element-wise).
  inline void mul(value_type* buffer) const {
    mul_(buffer, std::integral_constant<bool, is_special_multiplicable<E>::value>());  // NOLINT
  }

  // buffer /= e: evaluates the expression e, and divides the result by
  // the buffer (element-wise).
  inline void div(value_type* buffer) const {
    div_(buffer, std::integral_constant<bool, is_special_divisible<E>::value>());  // NOLINT
  }

 private:
  void assign_(value_type* buffer, std::true_type) const;
  void assign_(value_type* buffer, std::false_type) const;

  void add_(value_type* buffer, std::true_type) const;
  void add_(value_type* buffer, std::false_type) const;

  void sub_(value_type* buffer, std::true_type) const;
  void sub_(value_type* buffer, std::false_type) const;

  void mul_(value_type* buffer, std::true_type) const;
  void mul_(value_type* buffer, std::false_type) const;

  void div_(value_type* buffer, std::true_type) const;
  void div_(value_type* buffer, std::false_type) const;
};

template<typename E>
inline
void
expression_evaluator<E>::assign_(value_type* buffer, std::true_type) const {
  special_expression::assign(e, buffer);
}

template<typename E>
inline
void
expression_evaluator<E>::assign_(value_type* buffer, std::false_type) const {
  std::copy(e.begin(), e.end(), buffer);
}

template<typename E>
inline
void
expression_evaluator<E>::add_(value_type* buffer, std::true_type) const {
  special_expression::add(e, buffer);
}

template<typename E>
inline
void
expression_evaluator<E>::add_(value_type* buffer, std::false_type) const {
  std::for_each(e.begin(), e.end(),
                [&](const value_type& e) { *buffer++ += e; });
}

template<typename E>
inline
void
expression_evaluator<E>::sub_(value_type* buffer, std::true_type) const {
  special_expression::sub(e, buffer);
}

template<typename E>
inline
void
expression_evaluator<E>::sub_(value_type* buffer, std::false_type) const {
  std::for_each(e.begin(), e.end(),
                [&](const value_type& e) { *buffer++ -= e; });
}

template<typename E>
inline
void
expression_evaluator<E>::mul_(value_type* buffer, std::true_type) const {
  special_expression::mul(e, buffer);
}

template<typename E>
inline
void
expression_evaluator<E>::mul_(value_type* buffer, std::false_type) const {
  std::for_each(e.begin(), e.end(),
                [&](const value_type& e) { *buffer++ *= e; });
}

template<typename E>
inline
void
expression_evaluator<E>::div_(value_type* buffer, std::true_type) const {
  special_expression::div(e, buffer);
}

template<typename E>
inline
void
expression_evaluator<E>::div_(value_type* buffer, std::false_type) const {
  std::for_each(e.begin(), e.end(),
                [&](const value_type& e) { *buffer++ /= e; });
}
}  // namespace linalg_detail
}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_EXPRESSION_EVALUATOR_H_
