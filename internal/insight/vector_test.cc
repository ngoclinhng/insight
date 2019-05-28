// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <vector>

#include "insight/linalg/vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;
using ::testing::DoubleEq;

TEST(vector, empty_constructor) {
  vector<double> v;

  ASSERT_TRUE(v.empty());
  ASSERT_EQ(v.size(), 0);
  ASSERT_EQ(v.capacity(), 0);
}

TEST(vector, constructor_from_initializer_list) {
  column_vector<double> v = {1, 2, 3, 4};

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 4);
  ASSERT_EQ(v.capacity(), 4);
  ASSERT_EQ(v.num_rows(), 4);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 4);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(1, 2, 3, 4));

  row_vector<double> w = {1, 2, 3, 4};

  ASSERT_FALSE(w.empty());
  ASSERT_EQ(w.size(), 4);
  ASSERT_EQ(w.capacity(), 4);
  ASSERT_EQ(w.num_rows(), 1);
  ASSERT_EQ(w.num_cols(), 4);
  ASSERT_EQ(w.shape().first, 1);
  ASSERT_EQ(w.shape().second, 4);
  ASSERT_THAT(v, ElementsAre(1, 2, 3, 4));
}

TEST(vector, assignment_operator) {
  column_vector<double> v;
  column_vector<double> w = {10, 20, 30 , 40};
  column_vector<double> x = {4, 6};

  v = w;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 4);
  ASSERT_EQ(v.capacity(), 4);
  ASSERT_EQ(v.num_rows(), 4);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 4);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(10, 20, 30, 40));

  v = x;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v.capacity(), 4);
  ASSERT_EQ(v.num_rows(), 2);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 2);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(4, 6));

  v = {1, 2, 3, 4};

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 4);
  ASSERT_EQ(v.capacity(), 4);
  ASSERT_EQ(v.num_rows(), 4);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 4);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(1, 2, 3, 4));
}

TEST(vector, constructor_from_input_iterator_range) {
  std::vector<double> vec = {1, 2, 3, 4};
  vector<double> v(vec.begin(), vec.end());

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 4);
  ASSERT_EQ(v.capacity(), 4);
  ASSERT_EQ(v.num_rows(), 4);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 4);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(1, 2, 3, 4));

  double data[5] = {10, 20, 30, 40, 50};
  column_vector<double> w(data, data + 5);

  ASSERT_FALSE(w.empty());
  ASSERT_EQ(w.size(), 5);
  ASSERT_EQ(w.capacity(), 5);
  ASSERT_EQ(w.num_rows(), 5);
  ASSERT_EQ(w.num_cols(), 1);
  ASSERT_EQ(w.shape().first, 5);
  ASSERT_EQ(w.shape().second, 1);
  ASSERT_THAT(w, ElementsAre(10, 20, 30, 40, 50));
}

TEST(vector, operator_plus_equal_scalar) {
  vector<double> v = {1, 2, 3, 4, 5};
  v += 2.0;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(3, 4, 5, 6, 7));

  vector<int> w = {10, 20, 30, 40, 50};
  w += 2;

  ASSERT_FALSE(w.empty());
  ASSERT_EQ(w.size(), 5);
  ASSERT_EQ(w.capacity(), 5);
  ASSERT_EQ(w.num_rows(), 5);
  ASSERT_EQ(w.num_cols(), 1);
  ASSERT_EQ(w.shape().first, 5);
  ASSERT_EQ(w.shape().second, 1);
  ASSERT_THAT(w, ElementsAre(12, 22, 32, 42, 52));
}

TEST(vector, operator_minus_equal_scalar) {
  column_vector<double> v = {1, 2, 3, 4, 5};
  v -= 2.0;
  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(-1, 0, 1, 2, 3));
}

TEST(vector, operator_times_equal_scalar) {
  column_vector<double> v = {1, 2, 3, 4, 5};
  v *= 10.0;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(10, 20, 30, 40, 50));

  vector<int> w = {1, 2, 3, 4, 5};
  w *= 2;

  ASSERT_FALSE(w.empty());
  ASSERT_EQ(w.size(), 5);
  ASSERT_EQ(w.capacity(), 5);
  ASSERT_EQ(w.num_rows(), 5);
  ASSERT_EQ(w.num_cols(), 1);
  ASSERT_EQ(w.shape().first, 5);
  ASSERT_EQ(w.shape().second, 1);
  ASSERT_THAT(w, ElementsAre(2, 4, 6, 8, 10));
}

TEST(vector, operator_divides_equal_scalar) {
  column_vector<double> v = {10, 20, 30, 40, 50};
  v /= 10.0;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(1, 2, 3, 4, 5));

  column_vector<int> w = {2, 4, 6, 8, 10};
  w /= 2;

  ASSERT_FALSE(w.empty());
  ASSERT_EQ(w.size(), 5);
  ASSERT_EQ(w.capacity(), 5);
  ASSERT_EQ(w.num_rows(), 5);
  ASSERT_EQ(w.num_cols(), 1);
  ASSERT_EQ(w.shape().first, 5);
  ASSERT_EQ(w.shape().second, 1);
  ASSERT_THAT(w, ElementsAre(1, 2, 3, 4, 5));
}

TEST(vector, operator_plus_equal_other_vector) {
  column_vector<double> v = {1, 2, 3, 4, 5};
  column_vector<double> w = {10, 20, 30, 40, 50};

  v += w;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(11, 22, 33, 44, 55));

  v += v;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(22, 44, 66, 88, 110));

  column_vector<int> x = {1, 2, 3, 4, 5};
  column_vector<int> y = {1, 3, 5, 7, 9};

  x += y;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 5);
  ASSERT_EQ(x.capacity(), 5);
  ASSERT_EQ(x.num_rows(), 5);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 5);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(2, 5, 8, 11, 14));

  x += x;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 5);
  ASSERT_EQ(x.capacity(), 5);
  ASSERT_EQ(x.num_rows(), 5);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 5);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(4, 10, 16, 22, 28));
}

TEST(vector, operator_minus_equal_other_vector) {
  vector<double> v = {10, 20, 30, 40, 50};
  vector<double> w = {1, 2, 3, 4, 5};

  v -= w;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(9, 18, 27, 36, 45));

  v -= v;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(0, 0, 0, 0, 0));

  vector<int> x = {1, 2, 3, 4, 5};
  vector<int> y = {1, 3, 5, 7, 9};

  x -= y;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 5);
  ASSERT_EQ(x.capacity(), 5);
  ASSERT_EQ(x.num_rows(), 5);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 5);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(0, -1, -2, -3, -4));

  x -= x;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 5);
  ASSERT_EQ(x.capacity(), 5);
  ASSERT_EQ(x.num_rows(), 5);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 5);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(0, 0, 0, 0, 0));
}

TEST(vector, operator_times_equal_other) {
  vector<double> v = {1, 2, 3, 4, 5};
  vector<double> w = {1, 3, 5, 7, 9};

  v *= w;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(1, 6, 15, 28, 45));

  w *= w;

  ASSERT_FALSE(w.empty());
  ASSERT_EQ(w.size(), 5);
  ASSERT_EQ(w.capacity(), 5);
  ASSERT_EQ(w.num_rows(), 5);
  ASSERT_EQ(w.num_cols(), 1);
  ASSERT_EQ(w.shape().first, 5);
  ASSERT_EQ(w.shape().second, 1);
  ASSERT_THAT(w, ElementsAre(1, 9, 25, 49, 81));

  vector<int> x = {1, 2, 3, 4};
  vector<int> y = {1, 3, 5, 7};;

  x *= y;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 4);
  ASSERT_EQ(x.capacity(), 4);
  ASSERT_EQ(x.num_rows(), 4);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 4);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(1, 6, 15, 28));

  y *= y;

  ASSERT_FALSE(y.empty());
  ASSERT_EQ(y.size(), 4);
  ASSERT_EQ(y.capacity(), 4);
  ASSERT_EQ(y.num_rows(), 4);
  ASSERT_EQ(y.num_cols(), 1);
  ASSERT_EQ(y.shape().first, 4);
  ASSERT_EQ(y.shape().second, 1);
  ASSERT_THAT(y, ElementsAre(1, 9, 25, 49));
}

TEST(vector, operator_divides_equal_other_vector) {
  vector<double> v = {10, 20, 30, 40, 50};
  vector<double> w = {2, 10, 3, 8, 10};

  v /= w;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(5, 2, 10, 5, 5));

  v /= v;

  ASSERT_FALSE(v.empty());
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v.capacity(), 5);
  ASSERT_EQ(v.num_rows(), 5);
  ASSERT_EQ(v.num_cols(), 1);
  ASSERT_EQ(v.shape().first, 5);
  ASSERT_EQ(v.shape().second, 1);
  ASSERT_THAT(v, ElementsAre(1, 1, 1, 1, 1));

  vector<int> x = {10, 20, 30, 40};
  vector<int> y = {2, 10, 3, 8};

  x /= y;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 4);
  ASSERT_EQ(x.capacity(), 4);
  ASSERT_EQ(x.num_rows(), 4);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 4);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(5, 2, 10, 5));

  x /= x;

  ASSERT_FALSE(x.empty());
  ASSERT_EQ(x.size(), 4);
  ASSERT_EQ(x.capacity(), 4);
  ASSERT_EQ(x.num_rows(), 4);
  ASSERT_EQ(x.num_cols(), 1);
  ASSERT_EQ(x.shape().first, 4);
  ASSERT_EQ(x.shape().second, 1);
  ASSERT_THAT(x, ElementsAre(1, 1, 1, 1));
}

TEST(vector, norm2) {
  vector<double> v = {3, 4, 0};

  ASSERT_THAT(v.norm2(), DoubleEq(5));

  vector<int> x = {3, 4};

  ASSERT_EQ(x.norm2(), 5);
}

TEST(vector, dot) {
  vector<double> v = {1, 2, 3, 4, 5};
  vector<double> w = {0, 10, -3, 4, 20};

  ASSERT_THAT(v.dot(w), DoubleEq(127));
  ASSERT_THAT(w.dot(v), DoubleEq(127));

  vector<int> x = {-1, 2, 3};
  vector<int> y = {10, 20, 30};

  ASSERT_EQ(x.dot(y), 120);
  ASSERT_EQ(y.dot(x), 120);
}

}  // namespace insight
