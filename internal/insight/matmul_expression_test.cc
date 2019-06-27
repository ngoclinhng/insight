// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
#include "insight/linalg/vector.h"
#include "insight/linalg/functions.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(matmul_expression, int_matrix_mul_vector) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  vector<int> x = {-1, 0, 2};

  vector<int> y = matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 2);
  EXPECT_EQ(y.row_count(), 2);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 2);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(5, 8));

  y += matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 2);
  EXPECT_EQ(y.row_count(), 2);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 2);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(10, 16));

  y -= matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 2);
  EXPECT_EQ(y.row_count(), 2);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 2);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(5, 8));

  y *= matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 2);
  EXPECT_EQ(y.row_count(), 2);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 2);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(25, 64));

  y /= matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 2);
  EXPECT_EQ(y.row_count(), 2);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 2);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(5, 8));
}

TEST(matmul_expression, float_Ax) {
  matrix<double> A = {{0.5, 1.0, 1.5, 2.0},
                      {2.5, 3.0, 3.5, 4.0},
                      {4.5, 5.0, 5.5, 6.0}};
  vector<double> x = {-0.5, 1.0, 0.0, 2.0};

  vector<double> y = matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4.75, 9.75, 14.75));

  y += matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(9.5, 19.5, 29.5));

  y -= matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4.75, 9.75, 14.75));

  y *= matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(22.5625, 95.0625, 217.5625));

  y /= matmul(A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4.75, 9.75, 14.75));
}

TEST(matmul_expression, float_aAx) {
  matrix<double> A = {{0.5, 1.0, 1.5, 2.0},
                      {2.5, 3.0, 3.5, 4.0},
                      {4.5, 5.0, 5.5, 6.0}};
  vector<double> x = {0, -1, 0.5, 1};

  vector<double> y = matmul(A, 2.0 * x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(3.5, 5.5, 7.5));

  y += matmul(2.0 * A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(7, 11, 15));

  y -= matmul(A * 2.0, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(3.5, 5.5, 7.5));

  y *= matmul(A, 2.0 * x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(12.25, 30.25, 56.25));

  y /= matmul(2.0 * A, x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(3.5, 5.5, 7.5));
}

TEST(matmul_expression, float_aAbx) {
  matrix<double> A = {{0.5, 1.0, 1.5, 2.0},
                      {2.5, 3.0, 3.5, 4.0},
                      {4.5, 5.0, 5.5, 6.0}};
  vector<double> x = {0, -1, 0.5, 1};

  vector<double> y = matmul(4.0 * A, 0.5 * x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(3.5, 5.5, 7.5));

  y += matmul(0.5 * A, x * 4.0);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(7, 11, 15));

  y -= matmul(A * 4.0, x * 0.5);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(3.5, 5.5, 7.5));

  y *= matmul(A * 0.5, 4.0 * x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(12.25, 30.25, 56.25));

  y /= matmul(4.0 * A, 0.5 * x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(3.5, 5.5, 7.5));
}

TEST(matmul_expression, float_A_transpose_times_x) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  vector<double> x = {0, -2};

  vector<double> y = matmul(A.t(), x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(-4, -5, -6));

  y += matmul(A.t(), 2.0 * x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(-12, -15, -18));

  y -= matmul(A.t(), x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(-8, -10, -12));

  y *= matmul(A.t(), x * 0.5);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(16, 25, 36));

  y /= matmul(A.t(), x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(-4, -5, -6));
}



}  // namespace insight
