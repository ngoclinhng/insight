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

TEST(unary_expression, sqrt_of_a_vector) {
  vector<double> x = {0, 4.0, 2.25, 16, 25};
  vector<double> y = sqrt(x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 5);
  EXPECT_EQ(y.row_count(), 5);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 5);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(0, 2, 1.5, 4, 5));

  y += sqrt(x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 5);
  EXPECT_EQ(y.row_count(), 5);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 5);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(0, 4, 3, 8, 10));

  y -= sqrt(x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 5);
  EXPECT_EQ(y.row_count(), 5);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 5);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(0, 2, 1.5, 4, 5));

  y *= sqrt(x);

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 5);
  EXPECT_EQ(y.row_count(), 5);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 5);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(0, 4, 2.25, 16, 25));
}

TEST(unary_expression, sqrt_of_a_vector_transpose) {
  vector<double> x = {0, 4.0, 2.25, 16, 25};
  matrix<double> y = sqrt(x.t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 1);
  EXPECT_EQ(y.col_count(), 5);
  EXPECT_EQ(y.shape().first, 1);
  EXPECT_EQ(y.shape().second, 5);
  EXPECT_THAT(y, ElementsAre(0, 2, 1.5, 4, 5));

  y += sqrt(x.t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 1);
  EXPECT_EQ(y.col_count(), 5);
  EXPECT_EQ(y.shape().first, 1);
  EXPECT_EQ(y.shape().second, 5);
  EXPECT_THAT(y, ElementsAre(0, 4, 3, 8, 10));

  y -= sqrt(x.t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 1);
  EXPECT_EQ(y.col_count(), 5);
  EXPECT_EQ(y.shape().first, 1);
  EXPECT_EQ(y.shape().second, 5);
  EXPECT_THAT(y, ElementsAre(0, 2, 1.5, 4, 5));

  y *= sqrt(x.t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 1);
  EXPECT_EQ(y.col_count(), 5);
  EXPECT_EQ(y.shape().first, 1);
  EXPECT_EQ(y.shape().second, 5);
  EXPECT_THAT(y, ElementsAre(0, 4, 2.25, 16, 25));
}

TEST(unary_expression, sqrt_of_a_float_dense_matrix) {
  matrix<double> A = {{1, 4, 2.25}, {16, 6.25, 25}};

  matrix<double> B = sqrt(A);

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_THAT(B, ElementsAre(1, 2, 1.5, 4, 2.5, 5));

  B += sqrt(A);

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_THAT(B, ElementsAre(2, 4, 3, 8, 5, 10));

  B -= sqrt(A);

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_THAT(B, ElementsAre(1, 2, 1.5, 4, 2.5, 5));

  B *= sqrt(A);

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_THAT(B, ElementsAre(1, 4, 2.25, 16, 6.25, 25));

  B /= sqrt(A);

  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.row_count(), 2);
  EXPECT_EQ(B.col_count(), 3);
  EXPECT_EQ(B.shape().first, 2);
  EXPECT_EQ(B.shape().second, 3);
  EXPECT_EQ(B.size(), 6);
  EXPECT_THAT(B, ElementsAre(1, 2, 1.5, 4, 2.5, 5));
}

TEST(unary_expresison, sqrt_of_a_float_dense_row_view) {
  matrix<double> A = {{1, 4, 2.25}, {16, 6.25, 25}};

  matrix<double> x = sqrt(A.row_at(0));

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 3);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 3);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(1, 2, 1.5));

  x += sqrt(A.row_at(0));

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 3);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 3);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(2, 4, 3));

  x -= sqrt(A.row_at(0));

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 3);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 3);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(1, 2, 1.5));

  x *= sqrt(A.row_at(0));

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 3);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 3);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(1, 4, 2.25));

  x /= sqrt(A.row_at(0));

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 3);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 3);
  EXPECT_EQ(x.size(), 3);
  EXPECT_THAT(x, ElementsAre(1, 2, 1.5));
}

TEST(unary_expression, sqrt_of_a_float_dense_row_view_transpose) {
  matrix<double> A = {{1, 4, 2.25}, {16, 6.25, 25}};

  vector<double> y = sqrt(A.row_at(0).t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(1, 2, 1.5));

  y += sqrt(A.row_at(0).t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(2, 4, 3));

  y -= sqrt(A.row_at(0).t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(1, 2, 1.5));

  y *= sqrt(A.row_at(0).t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(1, 4, 2.25));

  y /= sqrt(A.row_at(0).t());

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.row_count(), 3);
  EXPECT_EQ(y.col_count(), 1);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(1, 2, 1.5));
}

}  // namespace insight
