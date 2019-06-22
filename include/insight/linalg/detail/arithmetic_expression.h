// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_DETAIL_ARITHMETIC_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_DETAIL_ARITHMETIC_EXPRESSION_H_

#include <functional>
#include <iterator>
#include <type_traits>

#include "insight/internal/unary_transform_iterator.h"
#include "insight/internal/binary_transform_iterator.h"

#include "glog/logging.h"

namespace insight {
namespace linalg_detail {
// Base class for all vector expressions.
template<typename Derived>
struct vector_expression {
  const Derived& self() const { return static_cast<const Derived&>(*this); }
};

// Base class for all matrix expressions.
template<typename Derived>
struct matrix_expression {
  const Derived& self() const { return static_cast<const Derived&>(*this); }
};

// Forward declarations
template<typename E> struct row_view;
template<typename E> struct col_view;
template<typename E> struct transpose_expression;

// Binary expression.

// Element-wise arithmetic (addition, substraction, multiplication, and
// division) between two generic vector/matrix expressions.
template<typename E1, typename E2, typename F>
struct binary_expression : public std::conditional<
  std::is_base_of<vector_expression<E1>, E1>::value &&
  std::is_base_of<vector_expression<E2>, E2>::value,
  vector_expression< binary_expression<E1, E2, F> >,
  matrix_expression< binary_expression<E1, E2, F> >
  >::type {
 private:
  using self = binary_expression<E1, E2, F>;

 public:
  using value_type = typename E1::value_type;
  using size_type = typename E1::size_type;
  using shape_type = typename E1::shape_type;
  using functor_type = F;
  using const_iterator =
      internal::binary_transform_iterator<typename E1::const_iterator,
                                          typename E2::const_iterator,
                                          functor_type>;
  using iterator = const_iterator;

  const E1& e1;
  const E2& e2;
  const F& f;

  binary_expression(const E1& e1, const E2& e2, const F& f)
      : e1(e1), e2(e2), f(f) {
    CHECK_EQ(e1.row_count(), e2.row_count());
    CHECK_EQ(e1.col_count(), e2.col_count());
  }

  inline size_type row_count() const { return e1.row_count(); }
  inline size_type col_count() const  { return e1.col_count(); }
  inline shape_type shape() const  { return e1.shape(); }
  inline size_type size() const  { return e1.size(); }

  // Accesses the row at the given index row_index. No bounds checking is
  // performed.
  inline row_view<self> row_at(size_type row_index) {
    return row_view<self>(this, row_index);
  }

  // Accesses the column at the given index col_index. No bounds checking is
  // performed.
  inline col_view<self> col_at(size_type col_index) {
    return col_view<self>(this, col_index);
  }

  // Return the transpose of this expression.
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  // Iterator.

  inline const_iterator begin() const {
    return internal::make_binary_transform_iterator(e1.cbegin(), e2.cbegin(),
                                                    f);
  }

  inline const_iterator cbegin() const {
    return internal::make_binary_transform_iterator(e1.cbegin(), e2.cbegin(),
                                                    f);
  }

  inline const_iterator end() const {
    return internal::make_binary_transform_iterator(e1.cend(), e2.end(), f);
  }

  inline const_iterator cend() const  {
    return internal::make_binary_transform_iterator(e1.cend(), e2.end(), f);
  }
};

// Element-wise arithmetic between a generic vector/matrix expression
// and a scalar.
template<typename E, typename F>
struct binary_expression<E, typename E::value_type, F>
    : public std::conditional<
  std::is_base_of<vector_expression<E>, E>::value,
  vector_expression<binary_expression<E, typename E::value_type, F> >,
  matrix_expression<binary_expression<E, typename E::value_type, F> >
  >::type {
 private:
  using self = binary_expression<E, typename E::value_type, F>;

 public:
  using value_type = typename E::value_type;
  using size_type = typename E::size_type;
  using shape_type = typename E::shape_type;
  using functor_type = F;

  const E& e;
  const value_type scalar;
  const F& f;

  binary_expression(const E& e, value_type scalar, const F& f)
      : e(e), scalar(scalar), f(f) {
  }

  inline size_type row_count() const { return e.row_count(); }
  inline size_type col_count() const { return e.col_count(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline row_view<self> row_at(size_type row_index) {
    return row_view<self>(this, row_index);
  }

  inline col_view<self> col_at(size_type col_index) {
    return col_view<self>(this, col_index);
  }

  // Transpose of this expression.
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  using const_iterator = internal::unary_transform_iterator<
    typename E::const_iterator,
    decltype(std::bind(f, std::placeholders::_1, scalar))>;
  using iterator = const_iterator;

  inline const_iterator begin() const {
    auto u = std::bind(f, std::placeholders::_1, scalar);
    return internal::make_unary_transform_iterator(e.cbegin(), u);
  }

  inline const_iterator cbegin() const {
    auto u = std::bind(f, std::placeholders::_1, scalar);
    return internal::make_unary_transform_iterator(e.cbegin(), u);
  }

  inline const_iterator end() const {
    auto u = std::bind(f, std::placeholders::_1, scalar);
    return internal::make_unary_transform_iterator(e.cend(), u);
  }

  inline const_iterator cend() const  {
    auto u = std::bind(f, std::placeholders::_1, scalar);
    return internal::make_unary_transform_iterator(e.cend(), u);
  }
};

// Element-wise arithmetic between a scalar and a generic vector/matrix
// expression.
template<typename E, typename F>
struct binary_expression<typename E::value_type, E, F>
    : public std::conditional<
  std::is_base_of<vector_expression<E>, E>::value,
  vector_expression<binary_expression<typename E::value_type, E, F> >,
  matrix_expression<binary_expression<typename E::value_type, E, F> >
  >::type {
 private:
  using self = binary_expression<typename E::value_type, E, F>;

 public:
  using value_type = typename E::value_type;
  using size_type = typename E::size_type;
  using shape_type = typename E::shape_type;
  using functor_type = F;

  const value_type scalar;
  const E& e;
  const F& f;

  binary_expression(value_type scalar, const E& e, const F& f)
      : scalar(scalar), e(e), f(f) {
  }

  inline size_type row_count() const { return e.row_count(); }
  inline size_type col_count() const { return e.col_count(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline row_view<self> row_at(size_type row_index) {
    return row_view<self>(this, row_index);
  }

  inline col_view<self> col_at(size_type col_index) {
    return col_view<self>(this, col_index);
  }

  // Transpose of this expression.
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  using const_iterator = internal::unary_transform_iterator<
    typename E::const_iterator,
    decltype(std::bind(f, scalar, std::placeholders::_1))>;
  using iterator = const_iterator;

  inline const_iterator begin() const {
    auto u = std::bind(f, scalar, std::placeholders::_1);
    return internal::make_unary_transform_iterator(e.cbegin(), u);
  }

  inline const_iterator cbegin() const {
    auto u = std::bind(f, scalar, std::placeholders::_1);
    return internal::make_unary_transform_iterator(e.cbegin(), u);
  }

  inline const_iterator end() const {
    auto u = std::bind(f, scalar, std::placeholders::_1);
    return internal::make_unary_transform_iterator(e.cend(), u);
  }

  inline const_iterator cend() const  {
    auto u = std::bind(f, scalar, std::placeholders::_1);
    return internal::make_unary_transform_iterator(e.cend(), u);
  }
};

// Unary expression.

template<typename E, typename F>
struct unary_expression
    : public std::conditional<
  std::is_base_of<vector_expression<E>, E>::value,
  vector_expression<unary_expression<E, F> >,
  matrix_expression<unary_expression<E, F> >
  >::type {
 private:
  using self = unary_expression<E, F>;

 public:
  using value_type = typename E::value_type;
  using size_type = typename E::size_type;
  using shape_type = typename E::shape_type;
  using functor_type = F;
  using const_iterator = internal::unary_transform_iterator<
    typename E::const_iterator, F>;
  using iterator = const_iterator;

  const E& e;
  const F& f;

  unary_expression(const E& e, const F& f) : e(e), f(f) {}

  inline size_type row_count() const { return e.row_count(); }
  inline size_type col_count() const { return e.col_count(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline row_view<self> row_at(size_type row_index) {
    return row_view<self>(this, row_index);
  }

  inline col_view<self> col_at(size_type col_index) {
    return col_view<self>(this, col_index);
  }

  // Transpose of this expression.
  inline transpose_expression<self> t() const {
    return transpose_expression<self>(*this);
  }

  inline const_iterator begin() const {
    return internal::make_unary_transform_iterator(e.cbegin(), f);
  }

  inline const_iterator cbegin() const {
    return internal::make_unary_transform_iterator(e.cbegin(), f);
  }

  inline const_iterator end() const {
    return internal::make_unary_transform_iterator(e.cend(), f);
  }

  inline const_iterator cend() const  {
    return internal::make_unary_transform_iterator(e.cend(), f);
  }
};

// overload operators for vector expressions.

// Addition between two generic vector expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for addition
// between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::plus<typename L::value_type> >
  >::type
operator+(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::plus<typename L::value_type>
    >(e1.self(), e2.self(), std::plus<typename L::value_type>());
}

// addition between a generic vector expression and a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
                  std::plus<typename E::value_type> >
operator+(const vector_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::plus<typename E::value_type>
    >(e.self(), scalar, std::plus<typename E::value_type>());
}

// addition between a scalar and a generic vector expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
                  std::plus<typename E::value_type> >
operator+(typename E::value_type scalar, const vector_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::plus<typename E::value_type>
    >(scalar, e.self(), std::plus<typename E::value_type>());
}

// Substraction between two generic vector expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for substraction
// between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::minus<typename L::value_type> >
  >::type
operator-(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::minus<typename L::value_type>
    >(e1.self(), e2.self(), std::minus<typename L::value_type>());
}

// Substraction between a generic vector expression and a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
                  std::minus<typename E::value_type> >
operator-(const vector_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::minus<typename E::value_type>
    >(e.self(), scalar, std::minus<typename E::value_type>());
}

// Substraction between a scalar and a generic vector expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
                  std::minus<typename E::value_type> >
operator-(typename E::value_type scalar, const vector_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::minus<typename E::value_type>
    >(scalar, e.self(), std::minus<typename E::value_type>());
}

// Element-wise multiplication between two generic vector
// expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// multiplication between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::multiplies<typename L::value_type> >
  >::type
operator*(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::multiplies<typename L::value_type>
    >(e1.self(), e2.self(), std::multiplies<typename L::value_type>());
}

// Multiplication between a generic vector expression and a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
            std::multiplies<typename E::value_type> >
operator*(const vector_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::multiplies<typename E::value_type>
    >(e.self(), scalar, std::multiplies<typename E::value_type>());
}

// Multiplication between a scalar and a generic vector expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
            std::multiplies<typename E::value_type> >
operator*(typename E::value_type scalar, const vector_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::multiplies<typename E::value_type>
    >(scalar, e.self(), std::multiplies<typename E::value_type>());
}

// Element-wise division between two generic vector expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// division between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::divides<typename L::value_type> >
  >::type
operator/(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::divides<typename L::value_type>
    >(e1.self(), e2.self(), std::divides<typename L::value_type>());
}

// Element-wise division between a generic vector expression and
// a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
                  std::divides<typename E::value_type> >
operator/(const vector_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::divides<typename E::value_type>
    >(e.self(), scalar, std::divides<typename E::value_type>());
}

// Element-wise division between a scalar and a generic vector
// expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
                  std::divides<typename E::value_type> >
operator/(typename E::value_type scalar, const vector_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::divides<typename E::value_type>
    >(scalar, e.self(), std::divides<typename E::value_type>());
}

// overload operators for matrix expressions.

// Addition between two generic matrix expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for addition
// between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::plus<typename L::value_type> >
  >::type
operator+(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::plus<typename L::value_type>
    >(e1.self(), e2.self(), std::plus<typename L::value_type>());
}

// addition between a generic matrix expression and a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
                  std::plus<typename E::value_type> >
operator+(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::plus<typename E::value_type>
    >(e.self(), scalar, std::plus<typename E::value_type>());
}

// addition between a scalar and a generic matrix expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
                  std::plus<typename E::value_type> >
operator+(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::plus<typename E::value_type>
    >(scalar, e.self(), std::plus<typename E::value_type>());
}

// Substraction between two generic matrix expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for substraction
// between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::minus<typename L::value_type> >
  >::type
operator-(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::minus<typename L::value_type>
    >(e1.self(), e2.self(), std::minus<typename L::value_type>());
}

// Substraction between a generic matrix expression and a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
                  std::minus<typename E::value_type> >
operator-(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::minus<typename E::value_type>
    >(e.self(), scalar, std::minus<typename E::value_type>());
}

// Substraction between a scalar and a generic matrix expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
                  std::minus<typename E::value_type> >
operator-(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::minus<typename E::value_type>
    >(scalar, e.self(), std::minus<typename E::value_type>());
}

// Element-wise multiplication between two generic vector
// expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// multiplication between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::multiplies<typename L::value_type> >
  >::type
operator*(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::multiplies<typename L::value_type>
    >(e1.self(), e2.self(), std::multiplies<typename L::value_type>());
}

// Multiplication between a generic matrix expression and a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
            std::multiplies<typename E::value_type> >
operator*(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::multiplies<typename E::value_type>
    >(e.self(), scalar, std::multiplies<typename E::value_type>());
}

// Multiplication between a scalar and a generic matrix expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
            std::multiplies<typename E::value_type> >
operator*(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::multiplies<typename E::value_type>
    >(scalar, e.self(), std::multiplies<typename E::value_type>());
}

// Element-wise division between two generic matrix expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// division between an integer matrix and a floating-point matrix.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  binary_expression<L, R, std::divides<typename L::value_type> >
  >::type
operator/(const matrix_expression<L>& e1, const matrix_expression<R>& e2) {
  return binary_expression<
    L, R,
    std::divides<typename L::value_type>
    >(e1.self(), e2.self(), std::divides<typename L::value_type>());
}

// Element-wise division between a generic matrix expression and
// a scalar.
template<typename E>
inline
binary_expression<E, typename E::value_type,
                  std::divides<typename E::value_type> >
operator/(const matrix_expression<E>& e, typename E::value_type scalar) {
  return binary_expression<
    E,
    typename E::value_type,
    std::divides<typename E::value_type>
    >(e.self(), scalar, std::divides<typename E::value_type>());
}

// Element-wise division between a scalar and a generic matrix expression.
template<typename E>
inline
binary_expression<typename E::value_type, E,
                  std::divides<typename E::value_type> >
operator/(typename E::value_type scalar, const matrix_expression<E>& e) {
  return binary_expression<
    typename E::value_type,
    E,
    std::divides<typename E::value_type>
    >(scalar, e.self(), std::divides<typename E::value_type>());
}

}  // namespace linalg_detail
}  // namespace insight

#endif  // INCLUDE_INSIGHT_LINALG_DETAIL_ARITHMETIC_EXPRESSION_H_
