// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/vector.h"
#include "insight/linalg/matrix.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(matrix_vector_multiplication, int_matrix_mul_vector) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{0, 2, 4}, {6, 8, 10}};
  vector<int> x = {1, 2, 3};

  vector<int> y = dot(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(14, 32));

  y = dot(A, x) + 1.0;

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(15, 33));

  y = dot(A + B, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));
}

TEST(matrix_vector_multiplication, float_matrix_mul_vector) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> B = {{0, 2, 4}, {6, 8, 10}};
  vector<double> x = {1, 2, 3};

  vector<double> y = dot(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(14, 32));

  y = dot(A, x) + 1.0;

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(15, 33));

  y = dot(A + B, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));

  y += dot(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(44, 116));

  y -= dot(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));

  y *= dot(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(420, 2688));

  y /= dot(A, x);

  EXPECT_EQ(y.num_rows(), 2);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 2);
  EXPECT_THAT(y, ElementsAre(30, 84));
}

TEST(matrix_vector_multiplication, float_alpha_A_x) {
  matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
  vector<double> x = {-2, 4};

  vector<double> y = 0.5 * dot(A, x);

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(3, 5, 7));
}

TEST(matrix_vector_multiplication,
     float_dense_matrix_with_row_view_transpose) {
  matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
  matrix<double> B = {{-1, 2}, {0, 5}};

  vector<double> x = dot(A, B.row_at(0).t());

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(3, 5, 7));
}

TEST(matrix_vector_multiplication,
     transposed_float_dense_matrix_and_vector) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  vector<double> x = {-1, 2};

  vector<double> y = dot(A.t(), x);

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(7, 8, 9));
}

TEST(matrix_vector_multiplication,
     transposed_int_dense_matrix_and_vector) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  vector<int> x = {-1, 2};

  vector<int> y = dot(A.t(), x);

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(7, 8, 9));
}

}  // namespace insight