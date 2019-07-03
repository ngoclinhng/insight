// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <utility>
#include <vector>

#include "insight/linalg/matrix.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;
using ::testing::DoubleEq;

TEST(matrix, default_constructor) {
  matrix<double> m;

  EXPECT_EQ(m.row_count(), 0);
  EXPECT_EQ(m.col_count(), 0);
  EXPECT_EQ(m.size(), 0);
  EXPECT_TRUE(m.empty());
}

TEST(matrix, default_constructed_with_given_shape) {
  matrix<double> m(2, 3);

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0, 0, 0, 0, 0, 0));

  matrix<int> A(std::make_pair(2, 3));

  EXPECT_EQ(A.row_count(), 2);
  EXPECT_EQ(A.col_count(), 3);
  EXPECT_EQ(A.shape().first, 2);
  EXPECT_EQ(A.shape().second, 3);
  EXPECT_EQ(A.size(), 6);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(matrix, copy_constructed_from_value) {
  matrix<double> m(2, 3, 0.5);

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 0.5, 0.5, 0.5, 0.5, 0.5));

  matrix<int> A(std::make_pair(2, 3), 10);

  EXPECT_EQ(A.row_count(), 2);
  EXPECT_EQ(A.col_count(), 3);
  EXPECT_EQ(A.shape().first, 2);
  EXPECT_EQ(A.shape().second, 3);
  EXPECT_EQ(A.size(), 6);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(10, 10, 10, 10, 10, 10));
}

TEST(matrix, construct_from_forward_range) {
  std::vector<double> vec;
  vec.push_back(0.5);
  vec.push_back(1.5);
  vec.push_back(2.0);

  matrix<double> m(vec.begin(), vec.end());

  EXPECT_EQ(m.row_count(), 1);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 1);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 3);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.5, 2.0));

  double data[4] = {1.0, 2.0, 3.0, 4.0};
  matrix<double> A(data, data + 4);

  EXPECT_EQ(A.row_count(), 1);
  EXPECT_EQ(A.col_count(), 4);
  EXPECT_EQ(A.shape().first, 1);
  EXPECT_EQ(A.shape().second, 4);
  EXPECT_EQ(A.size(), 4);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(1, 2, 3, 4));
}

TEST(matrix, copy_constructor) {
  std::vector<double> vec = {0.5, 1.5, 2.0, 3.5, 4.0, 6.0};
  matrix<double> m(vec.begin(), vec.end());
  m.reshape(2, 3);

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.5, 2.0, 3.5, 4.0, 6.0));

  matrix<double> m1 = m;

  EXPECT_EQ(m1.row_count(), 2);
  EXPECT_EQ(m1.col_count(), 3);
  EXPECT_EQ(m1.shape().first, 2);
  EXPECT_EQ(m1.shape().second, 3);
  EXPECT_EQ(m1.size(), 6);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(0.5, 1.5, 2.0, 3.5, 4.0, 6.0));
}

TEST(matrix, assignment_operator) {
  std::vector<double> vec = {0.5, 1.5, 2.0, 3.5, 4.0, 6.0};
  matrix<double> m1(vec.begin(), vec.end());
  m1.reshape(2, 3);

  matrix<double> m = m1;
  m.reshape(3, 2);

  matrix<double> m2(vec.begin(), vec.begin() + 4);
  m2.reshape(2, 2);

  // from bigger matrix to smaller one.
  m1 = m2;

  EXPECT_EQ(m1.row_count(), 2);
  EXPECT_EQ(m1.col_count(), 2);
  EXPECT_EQ(m1.shape().first, 2);
  EXPECT_EQ(m1.shape().second, 2);
  EXPECT_EQ(m1.size(), 4);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(0.5, 1.5, 2.0, 3.5));

  // From smaller matrix to bigger one.
  m2 = m;

  EXPECT_EQ(m2.row_count(), 3);
  EXPECT_EQ(m2.col_count(), 2);
  EXPECT_EQ(m2.shape().first, 3);
  EXPECT_EQ(m2.shape().second, 2);
  EXPECT_EQ(m2.size(), 6);
  EXPECT_FALSE(m2.empty());
  EXPECT_THAT(m2, ElementsAre(0.5, 1.5, 2.0, 3.5, 4.0, 6.0));
}

// Dummy function to test move constructor.
matrix<double> create_2x2_matrix() {
  std::vector<double> v = {0.5, 1.0, 1.5, 2.0};
  matrix<double> m(v.begin(), v.end());
  m.reshape(2, 2);
  return m;
}

TEST(matrix, move_constructor) {
  matrix<double> m = create_2x2_matrix();

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 2);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 2);
  EXPECT_EQ(m.size(), 4);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.0, 1.5, 2.0));
}

TEST(matrix, swap) {
  std::vector<double> vec = {0.5, 1.5, 2.0, 3.5, 4.0, 6.0};

  matrix<double> m1(vec.begin(), vec.end());
  m1.reshape(2, 3);

  matrix<double> m2(vec.begin(), vec.begin() + 4);
  m2.reshape(2, 2);

  std::swap(m1, m2);

  EXPECT_EQ(m1.row_count(), 2);
  EXPECT_EQ(m1.col_count(), 2);
  EXPECT_EQ(m1.shape().first, 2);
  EXPECT_EQ(m1.shape().second, 2);
  EXPECT_EQ(m1.size(), 4);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(0.5, 1.5, 2.0, 3.5));

  EXPECT_EQ(m2.row_count(), 2);
  EXPECT_EQ(m2.col_count(), 3);
  EXPECT_EQ(m2.shape().first, 2);
  EXPECT_EQ(m2.shape().second, 3);
  EXPECT_EQ(m2.size(), 6);
  EXPECT_FALSE(m2.empty());
  EXPECT_THAT(m2, ElementsAre(0.5, 1.5, 2.0, 3.5, 4.0, 6.0));
}

TEST(matrix, constructor_from_initializer_list) {
  matrix<double> m = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

  EXPECT_EQ(m.row_count(), 1);
  EXPECT_EQ(m.col_count(), 6);
  EXPECT_EQ(m.shape().first, 1);
  EXPECT_EQ(m.shape().second, 6);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));

  m.reshape(2, 3);

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));
}

TEST(matrix, constructor_from_nested_initializer_list) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));
}

TEST(matrix, operator_equal_nested_initializer_list) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};

  // assign to a smaller nested initializer_list
  m = {{1, 2}, {3, 4}};

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 2);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 2);
  EXPECT_EQ(m.size(), 4);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4));

  // assign to a equal sized nested initializer_list
  m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));

  // assign to a bigger nested initializer_list.
  m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}, {3.5, 4.0, 4.5}};

  EXPECT_EQ(m.row_count(), 3);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 3);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 9);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5));
}

TEST(matrix, clear) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};
  m.clear();

  EXPECT_EQ(m.row_count(), 0);
  EXPECT_EQ(m.col_count(), 0);
  EXPECT_EQ(m.shape().first, 0);
  EXPECT_EQ(m.shape().second, 0);
  EXPECT_EQ(m.size(), 0);
  EXPECT_TRUE(m.empty());
}

TEST(matrix, plus_scalar) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  m += 0.5;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(1.0, 1.5, 2.0, 2.5, 3.0, 3.5));

  matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  m1 += 10;

  EXPECT_EQ(m1.row_count(), 3);
  EXPECT_EQ(m1.col_count(), 3);
  EXPECT_EQ(m1.shape().first, 3);
  EXPECT_EQ(m1.shape().second, 3);
  EXPECT_EQ(m1.size(), 9);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(11, 12, 13, 14, 15, 16, 17, 18, 19));
}

TEST(matrix, sub_scalar) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  m -= 0.5;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.0, 0.5, 1.0, 1.5, 2.0, 2.5));

  matrix<int> m1 = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
  m1 -= 100;

  EXPECT_EQ(m1.row_count(), 3);
  EXPECT_EQ(m1.col_count(), 3);
  EXPECT_EQ(m1.shape().first, 3);
  EXPECT_EQ(m1.shape().second, 3);
  EXPECT_EQ(m1.size(), 9);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(-90, -80, -70, -60, -50, -40, -30, -20, -10));
}

TEST(matrix, mul_scalar) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  m *= 2.0;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));

  matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  m1 *= 10;

  EXPECT_EQ(m1.row_count(), 3);
  EXPECT_EQ(m1.col_count(), 3);
  EXPECT_EQ(m1.shape().first, 3);
  EXPECT_EQ(m1.shape().second, 3);
  EXPECT_EQ(m1.size(), 9);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(10, 20, 30, 40, 50, 60, 70, 80, 90));
}

TEST(matrix, div_scalar) {
  matrix<double> m = {{-5, 10, -15}, {20, 25, 30}};
  m /= 10.0;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(-0.5, 1.0, -1.5, 2.0, 2.5, 3.0));

  matrix<int> m1 = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
  m1 /= 10;

  EXPECT_EQ(m1.row_count(), 3);
  EXPECT_EQ(m1.col_count(), 3);
  EXPECT_EQ(m1.shape().first, 3);
  EXPECT_EQ(m1.shape().second, 3);
  EXPECT_EQ(m1.size(), 9);
  EXPECT_FALSE(m1.empty());
  EXPECT_THAT(m1, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(matrix, add_matrix) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> m1 = {{1, 2, 3}, {4, 5, 6}};

  m += m1;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(1.5, 3, 4.5, 6, 7.5, 9));

  matrix<int> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix<int> B = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};

  A += B;

  EXPECT_EQ(A.row_count(), 3);
  EXPECT_EQ(A.col_count(), 3);
  EXPECT_EQ(A.shape().first, 3);
  EXPECT_EQ(A.shape().second, 3);
  EXPECT_EQ(A.size(), 9);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(11, 22, 33, 44, 55, 66, 77, 88, 99));
}

TEST(matrix, sub_matrix) {
  matrix<double> m = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> m1 = {{1, 2, 3}, {4, 5, 6}};

  m -= m1;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(-0.5, -1.0, -1.5, -2, -2.5, -3));

  matrix<int> A = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
  matrix<int> B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  A -= B;

  EXPECT_EQ(A.row_count(), 3);
  EXPECT_EQ(A.col_count(), 3);
  EXPECT_EQ(A.shape().first, 3);
  EXPECT_EQ(A.shape().second, 3);
  EXPECT_EQ(A.size(), 9);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(9, 18, 27, 36, 45, 54, 63, 72, 81));
}

TEST(matrix, mul_matrix) {
  matrix<double> m = {{10, 20, 30}, {40, 50, 60}};
  matrix<double> m1 = {{0.1, 0.5, 0.7}, {0.2, -0.3, 0.4}};

  m *= m1;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(1, 10, 21, 8, -15, 24));

  matrix<int> A = {{-1, -2, -3}, {4, 5, 6}, {1, 2, 3}};
  matrix<int> B = {{3, 5, 7}, {0, 1, 2}, {-2, 3, 4}};

  A *= B;

  EXPECT_EQ(A.row_count(), 3);
  EXPECT_EQ(A.col_count(), 3);
  EXPECT_EQ(A.shape().first, 3);
  EXPECT_EQ(A.shape().second, 3);
  EXPECT_EQ(A.size(), 9);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(-3, -10, -21, 0, 5, 12, -2, 6, 12));
}

TEST(matrix, div_matrix) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> m1(m.shape(), 10.0);

  m /= m1;

  EXPECT_EQ(m.row_count(), 2);
  EXPECT_EQ(m.col_count(), 3);
  EXPECT_EQ(m.shape().first, 2);
  EXPECT_EQ(m.shape().second, 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_FALSE(m.empty());
  EXPECT_THAT(m, ElementsAre(0.1, 0.2, 0.3, 0.4, 0.5, 0.6));

  matrix<int> A = {{2, 4, 9}, {10, 15, 16}, {21, 36, 40}};
  matrix<int> B = {{2, 2, 3}, {2, 5, 8}, {3, 6, 10}};

  A /= B;

  EXPECT_EQ(A.row_count(), 3);
  EXPECT_EQ(A.col_count(), 3);
  EXPECT_EQ(A.shape().first, 3);
  EXPECT_EQ(A.shape().second, 3);
  EXPECT_EQ(A.size(), 9);
  EXPECT_FALSE(A.empty());
  EXPECT_THAT(A, ElementsAre(1, 2, 3, 5, 3, 2, 7, 6, 4));
}

}  // namespace insight
