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

TEST(mat_mul, int_matrix_mul_vector) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{0, 2, 4}, {6, 8, 10}};
  vector<int> x = {1, 2, 3};

  vector<int> y = mat_mul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(14, 32));

  y = mat_mul(A, x) + 1.0;

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(15, 33));

  y = mat_mul(A + B, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));
}

TEST(mat_mul, float_matrix_mul_vector) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> B = {{0, 2, 4}, {6, 8, 10}};
  vector<double> x = {1, 2, 3};

  vector<double> y = mat_mul(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(14, 32));

  y = mat_mul(A, x) + 1.0;

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(15, 33));

  y = mat_mul(A + B, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));
}

}  // namespace insight
