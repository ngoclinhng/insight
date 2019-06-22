// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

// test addition between a matrix and a scalar.

TEST(matrix_expression, int_matrix_plus_scalar) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = A + 10;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(11, 12, 13, 14, 15, 16));

  B += A + 10;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(22, 24, 26, 28, 30, 32));

  B -= A + 10;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(11, 12, 13, 14, 15, 16));

  B *= (A + 10);

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(121, 144, 169, 196, 225, 256));

  B /= (A + 10);

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(11, 12, 13, 14, 15, 16));
}

TEST(matrix_expression, int_scalar_plus_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = 10 + A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(11, 12, 13, 14, 15, 16));

  B += (10 + A);

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(22, 24, 26, 28, 30, 32));

  B -= (10 + A);

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(11, 12, 13, 14, 15, 16));

  B *= (10 + A);

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(121, 144, 169, 196, 225, 256));

  B /= (10 + A);

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(11, 12, 13, 14, 15, 16));
}

TEST(matrix_expression, float_matrix_plus_scalar) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = A + 0.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 1.5, 2, 2.5, 3, 3.5));

  B += A + 0.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 3, 4, 5, 6, 7));

  B -= A + 0.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 1.5, 2, 2.5, 3, 3.5));

  B *= A + 0.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2.25, 4, 6.25, 9, 12.25));

  B /= A + 0.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 1.5, 2, 2.5, 3, 3.5));
}

TEST(matrix_expression, float_scalar_plus_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = 0.5 + A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 1.5, 2, 2.5, 3, 3.5));

  B += 0.5 + A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 3, 4, 5, 6, 7));

  B -= 0.5 + A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 1.5, 2, 2.5, 3, 3.5));

  B *= 0.5 + A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2.25, 4, 6.25, 9, 12.25));

  B /= 0.5 + A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 1.5, 2, 2.5, 3, 3.5));
}

// test substraction between a matrix and a scalar.

TEST(matrix_expression, int_matrix_minus_scalar) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = A - 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-1, 0, 1, 2, 3, 4));

  B += A - 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-2, 0, 2, 4, 6, 8));

  B -= A - 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-1, 0, 1, 2, 3, 4));

  B *= A - 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 0, 1, 4, 9, 16));
}

TEST(matrix_expression, int_scalar_minus_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = 7 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(6, 5, 4, 3, 2, 1));

  B += 7 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(12, 10, 8, 6, 4, 2));

  B -= 7 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(6, 5, 4, 3, 2, 1));

  B *= 7 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(36, 25, 16, 9, 4, 1));

  B /= 7 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(6, 5, 4, 3, 2, 1));
}

TEST(matrix_expression, float_matrix_minus_scalar) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = A - 3.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-3, -2.5, -2, -1.5, -1, -0.5));

  B += A - 3.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-6, -5, -4, -3, -2, -1));

  B -= A - 3.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-3, -2.5, -2, -1.5, -1, -0.5));

  B *= A - 3.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(9, 6.25, 4, 2.25, 1, 0.25));

  B /= A - 3.5;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(-3, -2.5, -2, -1.5, -1, -0.5));
}

TEST(matrix_expression, float_scalar_minus_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = 3.5 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(3, 2.5, 2, 1.5, 1, 0.5));

  B += 3.5 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(6, 5, 4, 3, 2, 1));

  B -= 3.5 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(3, 2.5, 2, 1.5, 1, 0.5));

  B *= 3.5 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(9, 6.25, 4, 2.25, 1, 0.25));

  B /= 3.5 - A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(3, 2.5, 2, 1.5, 1, 0.5));
}

// test multiplication between a matrix and a scalar.

TEST(matrix_expression, int_matrix_times_scalar) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = A * 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));

  B += A * 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(4, 8, 12, 16, 20, 24));

  B -= A * 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));

  B *= A * 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(4, 16, 36, 64, 100, 144));

  B /= A * 2;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));
}

TEST(matrix_expression, int_scalar_times_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = 2 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));

  B += 2 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(4, 8, 12, 16, 20, 24));

  B -= 2 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));

  B *= 2 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(4, 16, 36, 64, 100, 144));

  B /= 2 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));
}

TEST(matrix_expression, float_matrix_times_scalar) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = A * 2.0;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2, 3, 4, 5, 6));

  B += A * 2.0;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));

  B -= A * 2.0;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2, 3, 4, 5, 6));

  B *= A * 2.0;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 4, 9, 16, 25, 36));

  B /= A * 2.0;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(matrix_expression, float_scalar_times_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = 2.0 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2, 3, 4, 5, 6));

  B += 2.0 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(2, 4, 6, 8, 10, 12));

  B -= 2.0 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2, 3, 4, 5, 6));

  B *= 2.0 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 4, 9, 16, 25, 36));

  B /= 2.0 * A;

  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_FALSE(B.empty());
  EXPECT_THAT(B, ElementsAre(1, 2, 3, 4, 5, 6));
}

// test addition between two matrix.

TEST(matrix_expression, int_matrix_plus_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{10, 20, 30}, {40, 50, 60}};

  matrix<int> C = A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(11, 22, 33, 44, 55, 66));

  C += A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(22, 44, 66, 88, 110, 132));

  C -= A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(11, 22, 33, 44, 55, 66));

  C *= A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(121, 484, 1089, 1936, 3025, 4356));

  C /= A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(11, 22, 33, 44, 55, 66));
}

TEST(matrix_expression, float_matrix_plus_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = {{1.0, -0.5, 0}, {2.0, 3.0, 1.5}};

  matrix<double> C = A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(1.5, 0.5, 1.5, 4.0, 5.5, 4.5));

  C += A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(3, 1, 3, 8, 11, 9));

  C -= A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(1.5, 0.5, 1.5, 4.0, 5.5, 4.5));

  C *= A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(2.25, 0.25, 2.25, 16, 30.25, 20.25));

  C /= A + B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(1.5, 0.5, 1.5, 4.0, 5.5, 4.5));
}

// test substraction between two matrices.

TEST(matrix_expression, int_matrix_minus_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{10, 20, 30}, {40, 50, 60}};

  matrix<int> C = A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-9, -18, -27, -36, -45, -54));

  C += A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-18, -36, -54, -72, -90, -108));

  C -= A -B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-9, -18, -27, -36, -45, -54));

  C *= A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(81, 324, 729, 1296, 2025, 2916));

  C /= A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-9, -18, -27, -36, -45, -54));
}

TEST(matrix_expression, float_matrix_minus_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = {{1.0, -0.5, 0}, {2.0, 3.0, 1.5}};

  matrix<double> C = A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-0.5, 1.5, 1.5, 0, -0.5, 1.5));

  C += A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-1, 3, 3, 0, -1, 3));

  C -= A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-0.5, 1.5, 1.5, 0, -0.5, 1.5));

  C *= A - B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(0.25, 2.25, 2.25, 0, 0.25, 2.25));

  C /= A - B + 1.0;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(0.5, 0.9, 0.9, 0, 0.5, 0.9));
}

// test element-wise multiplication between two matrices.

TEST(matrix_expression, int_matrix_times_matrix) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{-2, 3, 4}, {0, 4, 10}};

  matrix<int> C = A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-2, 6, 12, 0, 20, 60));

  C += A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-4, 12, 24, 0, 40, 120));

  C -= A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-2, 6, 12, 0, 20, 60));

  C *= A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(4, 36, 144, 0, 400, 3600));
}

TEST(matrix_expression, double_matrix_times_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = {{-1.0, 0.5, 2.0}, {0.5, 2.0, 2.0}};

  matrix<double> C = A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-0.5, 0.5, 3.0, 1.0, 5.0, 6.0));

  C += A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-1, 1, 6, 2, 10, 12));

  C -= A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-0.5, 0.5, 3.0, 1.0, 5.0, 6.0));

  C *= A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(0.25, 0.25, 9, 1, 25, 36));

  C /= A * B;

  EXPECT_EQ(C.row_count(), 2);
  EXPECT_EQ(C.col_count(), 3);
  EXPECT_EQ(C.shape().first, 2);
  EXPECT_EQ(C.shape().second, 3);
  EXPECT_EQ(C.size(), 6);
  EXPECT_FALSE(C.empty());
  EXPECT_THAT(C, ElementsAre(-0.5, 0.5, 3.0, 1.0, 5.0, 6.0));
}

// TODO(Linh): test element-wise division between two matrices.

}  // namespace insight
