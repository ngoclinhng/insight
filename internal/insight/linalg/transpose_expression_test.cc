// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
#include "insight/linalg/vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(transpose_expression, of_a_dense_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = A.t();

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.size(), 6);
  EXPECT_EQ(B.row_count(), 3);
  EXPECT_EQ(B.col_count(), 2);
  EXPECT_THAT(B, ElementsAre(1, 4, 2, 5, 3, 6));

  B += A.t();

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.size(), 6);
  EXPECT_EQ(B.row_count(), 3);
  EXPECT_EQ(B.col_count(), 2);
  EXPECT_THAT(B, ElementsAre(2, 8, 4, 10, 6, 12));

  B -= A.t();

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.size(), 6);
  EXPECT_EQ(B.row_count(), 3);
  EXPECT_EQ(B.col_count(), 2);
  EXPECT_THAT(B, ElementsAre(1, 4, 2, 5, 3, 6));

  B *= A.t();

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.size(), 6);
  EXPECT_EQ(B.row_count(), 3);
  EXPECT_EQ(B.col_count(), 2);
  EXPECT_THAT(B, ElementsAre(1, 16, 4, 25, 9, 36));

  B /= A.t();

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.size(), 6);
  EXPECT_EQ(B.row_count(), 3);
  EXPECT_EQ(B.col_count(), 2);
  EXPECT_THAT(B, ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(transpose_expression, of_a_dense_vector) {
  vector<int> x = {1, 2, 3, 4, 5, 6};  // 6x1
  matrix<int> y = x.t();  // 1x6

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 6);
  EXPECT_EQ(y.row_count(), 1);
  EXPECT_EQ(y.col_count(), 6);
  EXPECT_THAT(y, ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(transpose_expression, of_a_binary_matrix_expression) {
  matrix<int> A = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  matrix<int> B = {{10, 20, 30, 40}, {50, 60, 70, 80}};

  matrix<int> C = (A + B).t();

  EXPECT_FALSE(C.empty());
  EXPECT_EQ(C.size(), 8);
  EXPECT_EQ(C.row_count(), 4);
  EXPECT_EQ(C.col_count(), 2);
  EXPECT_EQ(C.shape().first, 4);
  EXPECT_EQ(C.shape().second, 2);
  EXPECT_THAT(C, ElementsAre(11, 55, 22, 66, 33, 77, 44, 88));
}

TEST(transpose_expresison, of_a_vector_expression) {
  vector<int> x = {1, 2, 3};
  vector<int> y = {10, 20, 30};

  matrix<int> z = (x + y).t();  // 1x3 mattrix.

  EXPECT_FALSE(z.empty());
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.row_count(), 1);
  EXPECT_EQ(z.col_count(), 3);
  EXPECT_EQ(z.shape().first, 1);
  EXPECT_EQ(z.shape().second, 3);
  EXPECT_THAT(z, ElementsAre(11, 22, 33));
}

TEST(transpose_expression, of_a_row_view) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  vector<double> x = A.row_at(0).t();  // 3x1

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.row_count(), 3);
  EXPECT_EQ(x.col_count(), 1);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(0.5, 1.0, 1.5));
}

TEST(transpose_expresion, of_a_col_view) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> x = A.col_at(1).t();  // 1x2

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 2);
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 2);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 2);
  EXPECT_THAT(x, ElementsAre(1.0, 2.5));
}

TEST(transpose_expression, matrix_addition) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};  // 2 x 3
  matrix<double> B = {{1, 2}, {3, 4}, {5, 6}};  // 3 x 2

  matrix<double> C = A.t() + B;

  EXPECT_FALSE(C.empty());
  EXPECT_EQ(C.size(), 6);
  EXPECT_EQ(C.row_count(), 3);
  EXPECT_EQ(C.col_count(), 2);
  EXPECT_EQ(C.shape().first, 3);
  EXPECT_EQ(C.shape().second, 2);
  EXPECT_THAT(C, ElementsAre(1.5, 4, 4, 6.5, 6.5, 9));

  C += A.t() + B;

  EXPECT_FALSE(C.empty());
  EXPECT_EQ(C.size(), 6);
  EXPECT_EQ(C.row_count(), 3);
  EXPECT_EQ(C.col_count(), 2);
  EXPECT_EQ(C.shape().first, 3);
  EXPECT_EQ(C.shape().second, 2);
  EXPECT_THAT(C, ElementsAre(3, 8, 8, 13, 13, 18));

  C -= A.t() + B;

  EXPECT_FALSE(C.empty());
  EXPECT_EQ(C.size(), 6);
  EXPECT_EQ(C.row_count(), 3);
  EXPECT_EQ(C.col_count(), 2);
  EXPECT_EQ(C.shape().first, 3);
  EXPECT_EQ(C.shape().second, 2);
  EXPECT_THAT(C, ElementsAre(1.5, 4, 4, 6.5, 6.5, 9));

  C *= A.t() + B;

  EXPECT_FALSE(C.empty());
  EXPECT_EQ(C.size(), 6);
  EXPECT_EQ(C.row_count(), 3);
  EXPECT_EQ(C.col_count(), 2);
  EXPECT_EQ(C.shape().first, 3);
  EXPECT_EQ(C.shape().second, 2);
  EXPECT_THAT(C, ElementsAre(2.25, 16, 16, 42.25, 42.25, 81));

  C /= A.t() + B;

  EXPECT_FALSE(C.empty());
  EXPECT_EQ(C.size(), 6);
  EXPECT_EQ(C.row_count(), 3);
  EXPECT_EQ(C.col_count(), 2);
  EXPECT_EQ(C.shape().first, 3);
  EXPECT_EQ(C.shape().second, 2);
  EXPECT_THAT(C, ElementsAre(1.5, 4, 4, 6.5, 6.5, 9));
}

TEST(transpose_expression, vector_addition) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};  // 2 x 3
  vector<double> x = {10, 20, 30};

  vector<double> y = x + A.row_at(0).t();

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(10.5, 21, 31.5));
}
}  // namespace insight
