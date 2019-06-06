// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_SPECIAL_DENSE_EXPRESSION_TRAITS_H_
#define INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_SPECIAL_DENSE_EXPRESSION_TRAITS_H_

#include <functional>

#include "insight/linalg/type_traits/is_dense_vector.h"
#include "insight/linalg/type_traits/is_dense_matrix.h"

#include "insight/linalg/arithmetic_expression.h"
#include "insight/linalg/dot_expression.h"
#include "insight/linalg/transpose_expression.h"
#include "insight/linalg/functions.h"

namespace insight {

// 1. is_ax: is a particular binary expression of the form `a * x` where
//    `a` is a floating-point scalar and `x` is floating-point, dense
//    matrix/vector?

template<typename E> struct is_ax: public std::false_type{};
template<typename E> struct is_ax<const E>: public is_ax<E>{};
template<typename E> struct is_ax<volatile E>: public is_ax<E>{};
template<typename E> struct is_ax<volatile const E>: public is_ax<E>{};

template<typename E, typename T>
struct is_ax<binary_expression<E, T, std::multiplies<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               (is_dense_vector<E>::value ||
                                is_dense_matrix<E>::value)),
                              std::true_type,
                              std::false_type>::type{};

template<typename E, typename T>
struct is_ax<binary_expression<T, E, std::multiplies<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               (is_dense_vector<E>::value ||
                                is_dense_matrix<E>::value)),
                              std::true_type,
                              std::false_type>::type{};

// 2. is_xda: Is a particular binary expression of the form `x / a` where
//    `a` is a floating-point scalar, and `x` is a floating-point, dense
//    matrix/vector?

template<typename E> struct is_xda: public std::false_type{};
template<typename E> struct is_xda<const E>: public is_xda<E>{};
template<typename E> struct is_xda<volatile E>: public is_xda<E>{};
template<typename E> struct is_xda<volatile const E>: public is_xda<E>{};

template<typename E, typename T>
struct is_xda<binary_expression<E, T, std::divides<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               (is_dense_vector<E>::value ||
                                is_dense_matrix<E>::value)),
                              std::true_type,
                              std::false_type>::type{};

// 3. is_xpy: Is a particular binary expression of the form `x + y` where
//    `x`, `y` are both either floating-point, dense matrices or vectors?

template<typename E> struct is_xpy: public std::false_type{};
template<typename E> struct is_xpy<const E>: public is_xpy<E>{};
template<typename E> struct is_xpy<volatile E>: public is_xpy<E>{};
template<typename E> struct is_xpy<volatile const E>: public is_xpy<E>{};

template<typename E1, typename E2, typename T>
struct is_xpy<binary_expression<E1, E2, std::plus<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               ((is_dense_vector<E1>::value &&
                                 is_dense_vector<E2>::value) ||
                                (is_dense_matrix<E1>::value &&
                                 is_dense_matrix<E2>::value))),
                              std::true_type,
                              std::false_type>::type{};

// 4. is_xmy: Is a particular binary expression of the form `x - y` where
//    `x`, `y` are either both floating-point, dense matrices or vectors?

template<typename E> struct is_xmy: public std::false_type{};
template<typename E> struct is_xmy<const E>: public is_xmy<E>{};
template<typename E> struct is_xmy<volatile E>: public is_xmy<E>{};
template<typename E> struct is_xmy<volatile const E>: public is_xmy<E>{};

template<typename E1, typename E2, typename T>
struct is_xmy<binary_expression<E1, E2, std::minus<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               ((is_dense_vector<E1>::value &&
                                 is_dense_vector<E2>::value) ||
                                (is_dense_matrix<E1>::value &&
                                 is_dense_matrix<E2>::value))),
                              std::true_type,
                              std::false_type>::type{};

// 5. is_xty: Is a particular binary expression of the form `x * y` where
//    `x` and `y` are either both floating-point, dense matrices or vectors?

template<typename E> struct is_xty: public std::false_type{};
template<typename E> struct is_xty<const E>: public is_xty<E>{};
template<typename E> struct is_xty<volatile E>: public is_xty<E>{};
template<typename E> struct is_xty<volatile const E>: public is_xty<E>{};

template<typename E1, typename E2, typename T>
struct is_xty<binary_expression<E1, E2, std::multiplies<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               ((is_dense_vector<E1>::value &&
                                 is_dense_vector<E2>::value) ||
                                (is_dense_matrix<E1>::value &&
                                 is_dense_matrix<E2>::value))),
                              std::true_type,
                              std::false_type>::type{};

// 6. is_xdy: Is a particular binary expression of the form `x / y` where
//    `x` and `y` are either both floating-point, dense matrices or vectors?

template<typename E> struct is_xdy: public std::false_type{};
template<typename E> struct is_xdy<const E>: public is_xdy<E>{};
template<typename E> struct is_xdy<volatile E>: public is_xdy<E>{};
template<typename E> struct is_xdy<volatile const E>: public is_xdy<E>{};

template<typename E1, typename E2, typename T>
struct is_xdy<binary_expression<E1, E2, std::divides<T> > >
    : public std::conditional<(std::is_floating_point<T>::value &&
                               ((is_dense_vector<E1>::value &&
                                 is_dense_vector<E2>::value) ||
                                (is_dense_matrix<E1>::value &&
                                 is_dense_matrix<E2>::value))),
                              std::true_type,
                              std::false_type>::type{};

// 7. is_sqrt_x: Is a particular unary expression of the form `sqrt(x)` where
//    `x` is either a floating-point, dense vector or a floating-point, dense
//     matrix?

template<typename E> struct is_sqrt_x: public std::false_type{};
template<typename E> struct is_sqrt_x<const E>: public is_sqrt_x<E>{};
template<typename E> struct is_sqrt_x<volatile E>: public is_sqrt_x<E>{};
template<typename E> struct is_sqrt_x<volatile const E>: public is_sqrt_x<E>{};

template<typename E, typename T>
struct is_sqrt_x<unary_expression<E, unary_functor::sqrt<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              (is_dense_vector<E>::value ||
                               is_dense_matrix<E>::value),
                              std::true_type,
                              std::false_type>::type{};

// 8. is_exp_x: Is a particular unary expression of the form `exp(x)` where
//    `x` is either a floating-point, dense vector or a floating-point, dense
//     matrix?

template<typename E> struct is_exp_x: public std::false_type{};
template<typename E> struct is_exp_x<const E>: public is_exp_x<E>{};
template<typename E> struct is_exp_x<volatile E>: public is_exp_x<E>{};
template<typename E> struct is_exp_x<volatile const E>: public is_exp_x<E>{};

template<typename E, typename T>
struct is_exp_x<unary_expression<E, unary_functor::exp<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              (is_dense_vector<E>::value ||
                               is_dense_matrix<E>::value),
                              std::true_type,
                              std::false_type>::type{};

// 9. is_log_x: Is a particular unary expression of the form `log(x)` where
//    `x` is either a floating-point, dense vector or a floating-point, dense
//     matrix?

template<typename E> struct is_log_x: public std::false_type{};
template<typename E> struct is_log_x<const E>: public is_log_x<E>{};
template<typename E> struct is_log_x<volatile E>: public is_log_x<E>{};
template<typename E> struct is_log_x<volatile const E>: public is_log_x<E>{};

template<typename E, typename T>
struct is_log_x<unary_expression<E, unary_functor::log<T> > >
    : public std::conditional<std::is_floating_point<T>::value &&
                              (is_dense_vector<E>::value ||
                               is_dense_matrix<E>::value),
                              std::true_type,
                              std::false_type>::type{};

// 10. is_Ax: Is a particular matrix-vector multiplication of the form
//     `A * x` where A is a floating-point, dense matrix and `x` is a
//      floating-point, dense vector?

template<typename E> struct is_Ax: public std::false_type{};
template<typename E> struct is_Ax<const E>: public is_Ax<E>{};
template<typename E> struct is_Ax<volatile E>: public is_Ax<E>{};
template<typename E> struct is_Ax<volatile const E>: public is_Ax<E>{};

template<typename M, typename V>
struct is_Ax<dot_expression<M, V> >
    : public std::conditional<is_dense_matrix<M>::value &&
                              is_dense_vector<V>::value &&
                              std::is_floating_point<typename V::value_type>::value,  // NOLINT
                              std::true_type,
                              std::false_type>::type{};

// 11: is_Atx: Is a particular matrix-vector multiplication of the form
//     `A' * x` where A is a floating-point, dense matrix and `x` is a
//      floating-point, dense vector?

template<typename E> struct is_Atx: public std::false_type{};
template<typename E> struct is_Atx<const E>: public is_Atx<E>{};
template<typename E> struct is_Atx<volatile E>: public is_Atx<E>{};
template<typename E> struct is_Atx<volatile const E>: public is_Atx<E>{};

template<typename M, typename V>
struct is_Atx<dot_expression<transpose_expression<M>, V> >
    : public std::conditional<is_dense_matrix<M>::value &&
                              is_dense_vector<V>::value &&
                              std::is_floating_point<typename V::value_type>::value,  // NOLINT
                              std::true_type,
                              std::false_type>::type{};

// Final: Is E a special dense expression?

template<typename E> struct is_special_dense_expression
    : public std::conditional<is_ax<E>::value ||
                              is_xda<E>::value ||
                              is_xpy<E>::value ||
                              is_xmy<E>::value ||
                              is_xty<E>::value ||
                              is_xdy<E>::value ||
                              is_sqrt_x<E>::value ||
                              is_exp_x<E>::value ||
                              is_log_x<E>::value ||
                              is_Ax<E>::value ||
                              is_Atx<E>::value,
                              std::true_type,
                              std::false_type>::type{};

template<typename E> struct is_special_dense_expression<const E>
    : public is_special_dense_expression<E>{};
template<typename E> struct is_special_dense_expression<volatile E>
    : public is_special_dense_expression<E>{};
template<typename E> struct is_special_dense_expression<volatile const E>
    : public is_special_dense_expression<E>{};

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_TYPE_TRAITS_SPECIAL_DENSE_EXPRESSION_TRAITS_H_
