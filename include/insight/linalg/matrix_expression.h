// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_MATRIX_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_MATRIX_EXPRESSION_H_

#include <functional>
#include <type_traits>

#include "glog/logging.h"


namespace insight {

// Base class for all matrix expressions.
template<typename Derived>
struct matrix_expression {
  const Derived& self() const { return static_cast<const Derived&>(*this); }
};

// Binary expression.

// Element-wise arithmetic between two generic matrix expressions.
template<typename E1, typename E2, typename Function>
struct binary_expr :
      public matrix_expression< binary_expr<E1, E2, Function> > {
  using size_type = typename E1::size_type;
  using value_type = typename E1::value_type;
  using shape_type = typename E1::shape_type;

  const E1& e1;
  const E2& e2;
  const Function& f;

  binary_expr(const E1& e1, const E2& e2, const Function& f)
      : e1(e1), e2(e2), f(f) {
    CHECK_EQ(e1.num_rows(), e2.num_rows());
    CHECK_EQ(e1.num_cols(), e2.num_cols());
  }

  inline size_type num_rows() const { return e1.num_rows(); }
  inline size_type num_cols() const { return e1.num_cols(); }
  inline shape_type shape() const { return e1.shape(); }
  inline size_type size() const { return e1.size(); }

  inline value_type operator[](size_type i) const {
    return f(e1[i], e2[i]);
  }
};

// Element-wise arithmetic between a generic matrix expression and a scalar.
template<typename E, typename Function>
struct binary_expr<E, typename E::value_type, Function>
    : public matrix_expression<binary_expr<E, typename E::value_type,
                                           Function> > {
  using size_type = typename E::size_type;
  using value_type = typename E::value_type;
  using shape_type = typename E::shape_type;

  const E& e;
  const value_type scalar;
  const Function& f;

  binary_expr(const E& e, value_type scalar, const Function& f)
      : e(e), scalar(scalar), f(f) {}

  inline size_type num_rows() const { return e.num_rows(); }
  inline size_type num_cols() const { return e.num_cols(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline value_type operator[](size_type i) const {
    return f(e[i], scalar);
  }
};

// Element-wise arithmetic between a scalar and a generic matrix expression.
template<typename E, typename Function>
struct binary_expr<typename E::value_type, E, Function>
    : public matrix_expression<binary_expr<typename E::value_type, E,
                                           Function> > {
  using size_type = typename E::size_type;
  using value_type = typename E::value_type;
  using shape_type = typename E::shape_type;

  const value_type scalar;
  const E& e;
  const Function& f;

  binary_expr(value_type scalar, const E& e, const Function& f)
      : scalar(scalar), e(e), f(f) {}

  inline size_type num_rows() const { return e.num_rows(); }
  inline size_type num_cols() const { return e.num_cols(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline value_type operator[](size_type i) const {
    return f(scalar, e[i]);
  }
};

// matrix multiplication expression between two generic matrix expressions.
// Technically speaking, this is a binary operation between two generic
// matrix expressions but the way matrix multiplication works is
// significantly different from that of other element-wise binary
// operations. That's the primary reason why we seperate them in the
// first place.
template<typename E1, typename E2>
struct matmul_expr :
      public matrix_expression< matmul_expr<E1, E2> > {
  using size_type = typename E1::size_type;
  using value_type = typename E1::value_type;
  using shape_type = typename E1::shape_type;

  const E1& e1;
  const E2& e2;

  matmul_expr(const E1& e1, const E2& e2)
      : e1(e1), e2(e2) {
    CHECK_EQ(e1.num_cols(), e2.num_rows());
  }

  inline size_type num_rows() const { return e1.num_rows(); }
  inline size_type num_cols() const { return e2.num_cols(); }
  inline shape_type shape() const {
    return shape_type(num_rows(), num_cols());
  }
  inline size_type size() const { return num_rows() * num_cols(); }

  inline value_type operator[](size_type i) const {
    return value_type();
  }
};

// Unary matrix expressions.

enum class unary_operator {
  sqrt
};

template<typename E, unary_operator op>
struct unary_expr :
      public matrix_expression< unary_expr<E, op> > {
  using size_type = typename E::size_type;
  using value_type = typename E::value_type;
  using shape_type = typename E::shape_type;

  const E& e;

  explicit unary_expr(const E& e) : e(e) {}

  inline size_type num_rows() const { return e.num_rows(); }
  inline size_type num_cols() const { return e.num_cols(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline value_type operator[](size_type i) const {
    // TODO(Linh): handle this later.
    return value_type();
  }
};

// overload operators.

template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for addition
// between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type,
               typename R::value_type>::value,
  binary_expr<L, R, std::plus<typename L::value_type> > >::type
operator+(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expr<
    L, R,
    std::plus<typename L::value_type>
    >(e1.self(), e2.self(), std::plus<typename L::value_type>());
}

template<typename E>
inline
binary_expr<E, typename E::value_type, std::plus<typename E::value_type> >
operator+(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expr<
    E,
    typename E::value_type,
    std::plus<typename E::value_type>
    >(e.self(), scalar, std::plus<typename E::value_type>());
}

template<typename E>
inline
binary_expr<typename E::value_type, E, std::plus<typename E::value_type> >
operator+(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expr<
    typename E::value_type,
    E,
    std::plus<typename E::value_type>
    >(scalar, e.self(), std::plus<typename E::value_type>());
}

template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for substraction
// between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type,
               typename R::value_type>::value,
  binary_expr<L, R, std::minus<typename L::value_type> > >::type
operator-(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expr<
    L, R,
    std::minus<typename L::value_type>
    >(e1.self(), e2.self(), std::minus<typename L::value_type>());
}

template<typename E>
inline
binary_expr<E, typename E::value_type, std::minus<typename E::value_type> >
operator-(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expr<
    E,
    typename E::value_type,
    std::minus<typename E::value_type>
    >(e.self(), scalar, std::minus<typename E::value_type>());
}

template<typename E>
inline
binary_expr<typename E::value_type, E, std::minus<typename E::value_type> >
operator-(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expr<
    typename E::value_type,
    E,
    std::minus<typename E::value_type>
    >(scalar, e.self(), std::minus<typename E::value_type>());
}

template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// multiplication between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type,
               typename R::value_type>::value,
  binary_expr<L, R, std::multiplies<typename L::value_type> > >::type
operator*(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expr<
    L, R,
    std::multiplies<typename L::value_type>
    >(e1.self(), e2.self(), std::multiplies<typename L::value_type>());
}

template<typename E>
inline
binary_expr<E, typename E::value_type,
            std::multiplies<typename E::value_type> >
operator*(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expr<
    E,
    typename E::value_type,
    std::multiplies<typename E::value_type>
    >(e.self(), scalar, std::multiplies<typename E::value_type>());
}

template<typename E>
inline
binary_expr<typename E::value_type, E,
            std::multiplies<typename E::value_type> >
operator*(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expr<
    typename E::value_type,
    E,
    std::multiplies<typename E::value_type>
    >(scalar, e.self(), std::multiplies<typename E::value_type>());
}


template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// division between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type,
               typename R::value_type>::value,
  binary_expr<L, R, std::divides<typename L::value_type> > >::type
operator/(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expr<
    L, R,
    std::divides<typename L::value_type>
    >(e1.self(), e2.self(), std::divides<typename L::value_type>());
}

template<typename E>
inline
binary_expr<E, typename E::value_type, std::divides<typename E::value_type> >
operator/(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expr<
    E,
    typename E::value_type,
    std::divides<typename E::value_type>
    >(e.self(), scalar, std::divides<typename E::value_type>());
}

template<typename E>
inline
binary_expr<typename E::value_type, E, std::divides<typename E::value_type> >
operator/(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expr<
    typename E::value_type,
    E,
    std::divides<typename E::value_type>
    >(scalar, e.self(), std::divides<typename E::value_type>());
}

// matrix multiplication function
template<typename E1, typename E2>
inline
// We need to make sure whoever E1 and E2 are they must have the same
// element type. So that, for example, no viable overload for matrix
// multiplication between an integer matrix and a floating-point
// matrix.
typename std::enable_if<std::is_same<typename E1::value_type,
                                     typename E2::value_type>::value,
                        matmul_expr<E1, E2> >::type
matmul(const matrix_expression<E1>& e1, const matrix_expression<E2>& e2) {
  return matmul_expr<E1, E2>(e1.self(), e2.self());
}

// unary functions

template<typename E>
inline
unary_expr<E, unary_operator::sqrt>
sqrt(const matrix_expression<E>& e) {
  return unary_expr<E, unary_operator::sqrt>(e.self());
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_MATRIX_EXPRESSION_H_
