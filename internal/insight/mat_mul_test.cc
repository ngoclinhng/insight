// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/vector.h"
#include "insight/linalg/matrix.h"
#include "insight/linalg/matrix_mul_vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(matmul, int_matrix_mul_vector) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{0, 2, 4}, {6, 8, 10}};
  vector<int> x = {1, 2, 3};

  vector<int> y = matmul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(14, 32));

  y = matmul(A, x) + 1.0;

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(15, 33));

  y = matmul(A + B, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));
}

TEST(matmul, float_matrix_mul_vector) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> B = {{0, 2, 4}, {6, 8, 10}};
  vector<double> x = {1, 2, 3};

  vector<double> y = matmul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(14, 32));

  y = matmul(A, x) + 1.0;

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(15, 33));

  y = matmul(A + B, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));

  y += matmul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(44, 116));

  y -= matmul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));

  y *= matmul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(420, 2688));

  y /= matmul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));
}

TEST(matmul, float_alpha_A_x) {
  matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
  vector<double> x = {-2, 4};

  vector<double> y = 0.5 * matmul(A, x);

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(3, 5, 7));
}

TEST(matmul, float_dense_matrix_with_row_view_transpose) {
  matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
  matrix<double> B = {{-1, 2}, {0, 5}};

  vector<double> x = matmul(A, B.row_at(0).t());

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(3, 5, 7));
}

}  // namespace insight
