// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(fmatrix, operator_plus_equal_scalar) {
  matrix<double> m = {{1, 2}, {3, 4}, {5, 6}};
  m += 10;
  m.row_at(0) += 10.0;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 3);
  ASSERT_EQ(m.num_cols(), 2);
  ASSERT_EQ(m.shape().first, 3);
  ASSERT_EQ(m.shape().second, 2);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(21, 22, 13, 14, 15, 16));
}

TEST(fmatrix, operator_minus_equal_scalar) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};
  m -= 1.0;
  m.row_at(1) += 2.0;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(0, 1, 2, 5, 6, 7));
}

TEST(fmatrix, operator_times_equal_scalar) {
  matrix<float> m = {{1, 2, 3}, {4, 5, 6}};
  m *= 2.0;
  m.row_at(0) *= 4.0;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(8, 16, 24, 8, 10, 12));
}

TEST(fmatrix, operator_div_equal_scalar) {
  matrix<float> m = {{10, 20, 30}, {40, 60, 80}};
  m /= 10.0;
  m.row_at(1) /= 2;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 2, 3, 4));
}

TEST(fmatrix, operator_plus_equal_matrix) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> m1(m.shape(), 10.0);

  m += m1;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(11, 12, 13, 14, 15, 16));

  m += m;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(22, 24, 26, 28, 30, 32));

  m.row_at(0) += m.row_at(1);

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(50, 54, 58, 28, 30, 32));

  matrix<double> m2 = {1, 2, 3};
  m2 += m.row_at(0);

  ASSERT_FALSE(m2.empty());
  ASSERT_EQ(m2.num_rows(), 1);
  ASSERT_EQ(m2.num_cols(), 3);
  ASSERT_EQ(m2.shape().first, 1);
  ASSERT_EQ(m2.shape().second, 3);
  ASSERT_EQ(m2.size(), 3);
  ASSERT_EQ(m2.capacity(), 3);
  EXPECT_THAT(m2, ElementsAre(51, 56, 61));
}


TEST(fmatrix, operator_minus_equal_matrix) {
  matrix<float> m = {{1, 2, 3}, {4, 5, 6}};
  matrix<float> m1 = {{10, 20, 30}, {40, 50, 60}};

  m -= m1;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(-9, -18, -27, -36, -45, -54));

  m -= m;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(fmatrix, operator_plus_equal_scalar_times_matrix) {
  matrix<double> m = {{1, 2}, {3, 4}, {5, 6}};
  matrix<double> m1 = {{10, 20}, {30, 40}, {50, 60}};

  m += 2.0 * m1;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 3);
  ASSERT_EQ(m.num_cols(), 2);
  ASSERT_EQ(m.shape().first, 3);
  ASSERT_EQ(m.shape().second, 2);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(21, 42, 63, 84, 105, 126));

  m = {1.5, 2.5};
  m += m1.row_at(0) * 0.5;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 1);
  ASSERT_EQ(m.num_cols(), 2);
  ASSERT_EQ(m.shape().first, 1);
  ASSERT_EQ(m.shape().second, 2);
  ASSERT_EQ(m.size(), 2);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(6.5, 12.5));

  m1.row_at(0) += 2.0 * m1.row_at(1);

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 3);
  ASSERT_EQ(m1.num_cols(), 2);
  ASSERT_EQ(m1.shape().first, 3);
  ASSERT_EQ(m1.shape().second, 2);
  ASSERT_EQ(m1.size(), 6);
  ASSERT_EQ(m1.capacity(), 6);
  EXPECT_THAT(m1, ElementsAre(70, 100, 30, 40, 50, 60));

  m1.row_at(1) += m1.row_at(1) * 3.0;

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 3);
  ASSERT_EQ(m1.num_cols(), 2);
  ASSERT_EQ(m1.shape().first, 3);
  ASSERT_EQ(m1.shape().second, 2);
  ASSERT_EQ(m1.size(), 6);
  ASSERT_EQ(m1.capacity(), 6);
  EXPECT_THAT(m1, ElementsAre(70, 100, 120, 160, 50, 60));
}

TEST(fmatrix, operator_minus_equal_scalar_times_matrix) {
  matrix<double> m = {{5, 10, 15}, {20, 25, 30}};
  matrix<double> m1 = {{10, 20, 30}, {40, 50, 60}};

  // m = m - 0.1 * m1.
  m -= 0.1 * m1;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(4, 8, 12, 16, 20, 24));

  m1.row_at(1) -= 2.0 * m1.row_at(0);

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 2);
  ASSERT_EQ(m1.num_cols(), 3);
  ASSERT_EQ(m1.shape().first, 2);
  ASSERT_EQ(m1.shape().second, 3);
  ASSERT_EQ(m1.size(), 6);
  ASSERT_EQ(m1.capacity(), 6);
  EXPECT_THAT(m1, ElementsAre(10, 20, 30, 20, 10, 0));

  m1.row_at(0) -= 1.0 * m1.row_at(0);

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 2);
  ASSERT_EQ(m1.num_cols(), 3);
  ASSERT_EQ(m1.shape().first, 2);
  ASSERT_EQ(m1.shape().second, 3);
  ASSERT_EQ(m1.size(), 6);
  ASSERT_EQ(m1.capacity(), 6);
  EXPECT_THAT(m1, ElementsAre(0, 0, 0, 20, 10, 0));
}

}  // namespace insight
