// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/transpose_expression.h"
#include "insight/linalg/matrix.h"
#include "insight/linalg/vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {
using ::testing::ElementsAre;

TEST(transpose_expression, transpose_a_matrix) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> B = A.t();

  EXPECT_EQ(B.num_rows(), 3);
  EXPECT_EQ(B.num_cols(), 2);
  EXPECT_EQ(B.size(), 6);
  EXPECT_THAT(B, ElementsAre(1, 4, 2, 5, 3, 6));
  EXPECT_THAT(B.row_at(0), ElementsAre(1, 4));
  EXPECT_THAT(B.row_at(1), ElementsAre(2, 5));
  EXPECT_THAT(B.row_at(2), ElementsAre(3, 6));
}

TEST(transpose_expression, transpose_a_vector) {
  vector<double> x = {1, 2, 3};
  matrix<double> y = x.t();

  EXPECT_EQ(y.num_rows(), 1);
  EXPECT_EQ(y.num_cols(), 3);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(1, 2, 3));
  EXPECT_THAT(y.row_at(0), ElementsAre(1, 2, 3));
}

TEST(transpose_expression, transpose_an_expression) {
  matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
  matrix<double> B = {{10, 20}, {30, 40}, {50, 60}};
  matrix<double> C = (A + B).t();

  EXPECT_EQ(C.num_rows(), 2);
  EXPECT_EQ(C.num_cols(), 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_THAT(C, ElementsAre(11, 33, 55, 22, 44, 66));
}

TEST(transpose_expression, transposed_matrix_mul_vector) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  vector<double> x = {-2, 1};

  vector<double> y = matmul(A.t(), x);

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(2, 1, 0));

  vector<double> z = {10, 20, 30};
  z += matmul(A.t(), x);

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_THAT(z, ElementsAre(12, 21, 30));

  z -= matmul(A.t(), x);

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_THAT(z, ElementsAre(10, 20, 30));
}

TEST(transpose, of_a_row_view_of_a_dense_matrix) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};
  auto e = m.row_at(0);
  auto et = e.t();

  EXPECT_EQ(et.num_rows(), 3);
  EXPECT_EQ(et.num_cols(), 1);
  EXPECT_EQ(et.size(), 3);
  EXPECT_THAT(et, ElementsAre(1, 2, 3));
}
}  // namespace insight
