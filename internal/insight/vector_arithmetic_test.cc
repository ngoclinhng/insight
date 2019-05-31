// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(vector_arithmetic, float_vector_times_scalar) {
  vector<double> x = {10, 20, 30};
  vector<double> y = 0.1 * x;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(1, 2, 3));

  x += 2.0 * y;

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(12, 24, 36));

  x -= 3.0 * y;

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(9, 18, 27));

  x /= 3.0 * y;

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(3, 3, 3));

  x *= y * 2.0;

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(6, 12, 18));
}

TEST(vector_arithmetic, int_vector_times_scalar) {
  vector<int> x = {1, 2, 3};
  vector<int> y = 2 * x;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(2, 4, 6));

  x += 3 * y;

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(7, 14, 21));

  x -= 2 * y;

  EXPECT_EQ(x.num_rows(), 3);
  EXPECT_EQ(x.num_cols(), 1);
  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.shape().first, 3);
  EXPECT_EQ(x.shape().second, 1);
  EXPECT_THAT(x, ElementsAre(3, 6, 9));
}

TEST(vector_arithmetic, float_vector_div_scalar) {
  vector<double> x = {10, 20, 30};
  vector<double> y = x / 10.0;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(1, 2, 3));

  y += x/2.0;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(6, 12, 18));

  y -= x/5.0;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4, 8, 12));

  y *= x/10.0;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4, 16, 36));

  y /= x/10;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4, 8, 12));
}

TEST(vector_arithmetic, int_vector_div_scalar) {
  vector<int> x = {10, 20, 30};
  vector<int> y = x / 10;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(1, 2, 3));

  y += x/2;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(6, 12, 18));

  y -= x/5;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4, 8, 12));

  y *= x/10;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4, 16, 36));

  y /= x/10;

  EXPECT_EQ(y.num_rows(), 3);
  EXPECT_EQ(y.num_cols(), 1);
  EXPECT_EQ(y.size(), 3);
  EXPECT_EQ(y.shape().first, 3);
  EXPECT_EQ(y.shape().second, 1);
  EXPECT_THAT(y, ElementsAre(4, 8, 12));
}

}  // namespace insight
