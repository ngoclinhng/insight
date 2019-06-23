// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
#include "insight/linalg/vector.h"
#include "insight/linalg/functions.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using linalg_detail::expression_traits;

TEST(expression_traits, ax) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  vector<double> x = {1, 2, 3};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A * 2.0)>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(-3.0 * A)>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x * 7.0)>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(2.5 * x)>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) * 2.0)>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(8.5 * A.row_at(0))>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(0.5 * x.t())>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() * 2.5)>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(0.5 * A.row_at(0).t())>::category,
               linalg_detail::expression_category::ax>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0).t() * 2.4)>::category,
               linalg_detail::expression_category::ax>::value));
}

TEST(expression_traits, xpy) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B(A.shape(), 10.5);
  matrix<double> C = {-1.5, 2.6, 3.4};

  vector<double> x = {1, 2, 3};
  vector<double> y = {0.5, -1.5, 3.7};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A + B)>::category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x + y)>::category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) + B.row_at(0))>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) + A.row_at(1))>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) + C)>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C + A.row_at(0))>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() + y.t())>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() + C)>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C + y.t())>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() + A.row_at(0))>::
               category,
               linalg_detail::expression_category::xpy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) + x.t())>::
               category,
               linalg_detail::expression_category::xpy>::value));
}

TEST(expression_traits, xmy) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B(A.shape(), 10.5);
  matrix<double> C = {-1.5, 2.6, 3.4};

  vector<double> x = {1, 2, 3};
  vector<double> y = {0.5, -1.5, 3.7};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A - B)>::category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x - y)>::category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) - B.row_at(0))>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) - A.row_at(1))>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) - C)>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C - A.row_at(0))>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() - y.t())>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() - C)>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C - y.t())>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() - A.row_at(0))>::
               category,
               linalg_detail::expression_category::xmy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) - x.t())>::
               category,
               linalg_detail::expression_category::xmy>::value));
}

TEST(expression_traits, xty) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B(A.shape(), 10.5);
  matrix<double> C = {-1.5, 2.6, 3.4};

  vector<double> x = {1, 2, 3};
  vector<double> y = {0.5, -1.5, 3.7};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A * B)>::category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x * y)>::category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) * B.row_at(0))>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) * A.row_at(1))>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) * C)>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C * A.row_at(0))>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() * y.t())>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() * C)>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C * y.t())>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() * A.row_at(0))>::
               category,
               linalg_detail::expression_category::xty>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) * x.t())>::
               category,
               linalg_detail::expression_category::xty>::value));
}

TEST(expression_traits, xdy) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B(A.shape(), 10.5);
  matrix<double> C = {-1.5, 2.6, 3.4};

  vector<double> x = {1, 2, 3};
  vector<double> y = {0.5, -1.5, 3.7};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A / B)>::category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x / y)>::category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) / B.row_at(0))>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) / A.row_at(1))>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) / C)>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C / A.row_at(0))>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() / y.t())>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() / C)>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(C / y.t())>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(x.t() / A.row_at(0))>::
               category,
               linalg_detail::expression_category::xdy>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(A.row_at(0) / x.t())>::
               category,
               linalg_detail::expression_category::xdy>::value));
}

// test expression traits for specific unary functions

TEST(expression_traits, sqrt) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  vector<double> x = {1.3, 5.6};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(sqrt(A))>::
               category,
               linalg_detail::expression_category::sqrt>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(sqrt(x))>::
               category,
               linalg_detail::expression_category::sqrt>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(sqrt(A.row_at(0)))>::
               category,
               linalg_detail::expression_category::sqrt>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(sqrt(A.row_at(0).t()))>::
               category,
               linalg_detail::expression_category::sqrt>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(sqrt(x.t()))>::
               category,
               linalg_detail::expression_category::sqrt>::value));
}

TEST(expression_traits, exp) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  vector<double> x = {1.3, 5.6};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(exp(A))>::
               category,
               linalg_detail::expression_category::exp>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(exp(x))>::
               category,
               linalg_detail::expression_category::exp>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(exp(A.row_at(0)))>::
               category,
               linalg_detail::expression_category::exp>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(exp(A.row_at(0).t()))>::
               category,
               linalg_detail::expression_category::exp>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(exp(x.t()))>::
               category,
               linalg_detail::expression_category::exp>::value));
}


TEST(expression_traits, log) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  vector<double> x = {1.3, 5.6};

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(log(A))>::
               category,
               linalg_detail::expression_category::log>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(log(x))>::
               category,
               linalg_detail::expression_category::log>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(log(A.row_at(0)))>::
               category,
               linalg_detail::expression_category::log>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(log(A.row_at(0).t()))>::
               category,
               linalg_detail::expression_category::log>::value));

  EXPECT_TRUE((std::is_same<
               expression_traits<decltype(log(x.t()))>::
               category,
               linalg_detail::expression_category::log>::value));
}
}  // namespace insight
