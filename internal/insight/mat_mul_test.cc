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

// We only optimize for things that make sense.
//
// a * X, X * a -> is_sX.
// X/a          -> is_X_div_alpha
//
// X + Y -> is_X_plus_Y.
// X - Y -> is_X_minus_Y.  ===> is_X_op_Y.
// X * Y -> is_X_times_Y.
// X / Y -> is_X_div_Y.
//
// aX + Y -> is_alpha_X_plus_Y
// Y + aX -> is_Y_plus_alpha_X.
//
// aX - Y -> is_aX_minus_Y
// Y - aX -> is_Y_minus_aX.
//
// aX + bY -> is_aX_plus_bY
// aX - bY -> is_aX_minus_bY.
//
// Au -> is_Au. is_aX
// a * Ax, Ax * a
//
// Ax + y -> is_Ax_plus_y.
// y + Ax -> is_y_plus_Ax.
//
// Ax - y -> is_
// y - Ax                  => is_gemv
//
// a * Ax + y
// y + a * Ax
//
// a * Ax - y
// y - a * Ax
//
// a * Ax + b * y
// b * y + a * Ax


}  // namespace insight
