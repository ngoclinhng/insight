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

TEST(vector_arithmetic, float_vector_addition) {
  vector<double> x = {1, 2, 3};
  vector<double> y = {2, 4, 6};
  vector<double> z = x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(3, 6, 9));

  z += x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(6, 12, 18));

  z -= x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(3, 6, 9));

  z *= x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(9, 36, 81));

  z /= x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(3, 6, 9));
}

TEST(vector_arithmetic, int_vector_addition) {
  vector<int> x = {1, 2, 3};
  vector<int> y = {2, 4, 6};
  vector<int> z = x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(3, 6, 9));

  z += x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(6, 12, 18));

  z -= x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(3, 6, 9));

  z *= x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(9, 36, 81));

  z /= x + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(3, 6, 9));
}

TEST(vector_arithmetic, float_vector_substraction) {
  vector<double> x = {2, 4, 6};
  vector<double> y = {1, 2, 3};
  vector<double> z = x - y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(1, 2, 3));

  z -= 2.0 * (x - y);

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(-1, -2, -3));

  z += (x - y) * 3.0;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(2, 4, 6));

  z /= (x - y);

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(2, 2, 2));

  z *= (x - y) + 1.0;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(4, 6, 8));
}

TEST(vector_arithmetic, int_vector_substraction) {
  vector<int> x = {2, 4, 6};
  vector<int> y = {1, 2, 3};
  vector<int> z = x - y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(1, 2, 3));

  z -= 2 * (x - y);

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(-1, -2, -3));

  z += (x - y) * 3;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(2, 4, 6));

  z /= (x - y);

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(2, 2, 2));

  z *= (x - y) + 1;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(4, 6, 8));
}

TEST(vector_arithmetic, float_vector_elemwise_multiplication) {
  vector<double> x = {1, 2, 3};
  vector<double> y = {2, 4, 6};
  vector<double> z = x * y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(2, 8, 18));

  z += x*y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(4, 16, 36));

  z -= 3.0 * x * y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(-2, -8, -18));

  z /= x * y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(-1, -1, -1));
}

TEST(vector_arithmetic, int_vector_elemwise_multiplication) {
  vector<int> x = {1, 2, 3};
  vector<int> y = {2, 4, 6};
  vector<int> z = x * y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(2, 8, 18));

  z += x*y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(4, 16, 36));

  z -= 3 * x * y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(-2, -8, -18));

  z /= x * y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(-1, -1, -1));
}

TEST(vector_arithmetic, float_vector_elemwise_division) {
  vector<double> x = {10, 20, 30};
  vector<double> y = {1, 2, 3};
  vector<double> z = x/y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(10, 10, 10));

  z += x/y + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(21, 22, 23));
}

TEST(vector_arithmetic, int_vector_elemwise_division) {
  vector<int> x = {10, 20, 30};
  vector<int> y = {1, 2, 3};
  vector<int> z = x/y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(10, 10, 10));

  z += x/y + y;

  EXPECT_EQ(z.num_rows(), 3);
  EXPECT_EQ(z.num_cols(), 1);
  EXPECT_EQ(z.size(), 3);
  EXPECT_EQ(z.shape().first, 3);
  EXPECT_EQ(z.shape().second, 1);
  EXPECT_THAT(z, ElementsAre(21, 22, 23));
}
}  // namespace insight
