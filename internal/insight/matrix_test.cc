// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
// #include "insight/linalg/evaluate.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;
using ::testing::DoubleEq;

TEST(matrix, default_constructor) {
  matrix<double> m;

  ASSERT_TRUE(m.empty());
  ASSERT_EQ(m.num_rows(), 0);
  ASSERT_EQ(m.num_cols(), 0);
  ASSERT_EQ(m.size(), 0);
  ASSERT_EQ(m.capacity(), 0);
  ASSERT_EQ(m.shape().first, 0);
  ASSERT_EQ(m.shape().second, 0);
}

TEST(matrix, constructor_from_num_rows_and_num_cols) {
  matrix<double> m(2, 3);

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);

  EXPECT_THAT(m, ElementsAre(0, 0, 0, 0, 0, 0));

  matrix<double> m1(2, 1, 10.0);

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 2);
  ASSERT_EQ(m1.num_cols(), 1);
  ASSERT_EQ(m1.size(), 2);
  ASSERT_EQ(m1.capacity(), 2);
  ASSERT_EQ(m1.shape().first, 2);
  ASSERT_EQ(m1.shape().second, 1);

  EXPECT_THAT(m1, ElementsAre(10, 10));
}

TEST(matrix, copy_constructor) {
  matrix<double> A(2, 3, 5.0);
  matrix<double> B = A;

  ASSERT_FALSE(B.empty());
  ASSERT_EQ(B.num_rows(), 2);
  ASSERT_EQ(B.num_cols(), 3);
  ASSERT_EQ(B.shape().first, 2);
  ASSERT_EQ(B.shape().second, 3);
  ASSERT_EQ(B.size(), 6);
  ASSERT_EQ(B.capacity(), 6);
  EXPECT_THAT(B, ElementsAre(5, 5, 5, 5, 5, 5));

  B(0, 2) = 10.0;
  B(1, 1) = 6.0;

  EXPECT_THAT(B, ElementsAre(5, 5, 10, 5, 6, 5));
  EXPECT_THAT(A, ElementsAre(5, 5, 5, 5, 5, 5));
}

TEST(matrix, assignment_operator) {
  matrix<double> A(3, 2, 4.0);
  matrix<double> B(2, 2, 6.0);

  matrix<double> C = A;

  ASSERT_FALSE(C.empty());
  ASSERT_EQ(C.num_rows(), 3);
  ASSERT_EQ(C.num_cols(), 2);
  ASSERT_EQ(C.shape().first, 3);
  ASSERT_EQ(C.shape().second, 2);
  ASSERT_EQ(C.size(), 6);
  ASSERT_EQ(C.capacity(), 6);
  EXPECT_THAT(C, ElementsAre(4, 4, 4, 4, 4, 4));

  C = B;

  ASSERT_FALSE(C.empty());
  ASSERT_EQ(C.num_rows(), 2);
  ASSERT_EQ(C.num_cols(), 2);
  ASSERT_EQ(C.shape().first, 2);
  ASSERT_EQ(C.shape().second, 2);
  ASSERT_EQ(C.size(), 4);
  ASSERT_EQ(C.capacity(), 6);
  EXPECT_THAT(C, ElementsAre(6, 6, 6, 6));

  matrix<double> D;
  D = A;

  ASSERT_FALSE(D.empty());
  ASSERT_EQ(D.num_rows(), 3);
  ASSERT_EQ(D.num_cols(), 2);
  ASSERT_EQ(D.shape().first, 3);
  ASSERT_EQ(D.shape().second, 2);
  ASSERT_EQ(D.size(), 6);
  ASSERT_EQ(D.capacity(), 6);
  EXPECT_THAT(D, ElementsAre(4, 4, 4, 4, 4, 4));
}

TEST(matrix, constructor_from_initializer_list) {
  matrix<double> m = {10, 20, 30, 40};

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 1);
  ASSERT_EQ(m.num_cols(), 4);
  ASSERT_EQ(m.shape().first, 1);
  ASSERT_EQ(m.shape().second, 4);
  ASSERT_EQ(m.size(), 4);
  ASSERT_EQ(m.capacity(), 4);
  EXPECT_THAT(m, ElementsAre(10, 20, 30, 40));
}

TEST(matrix, assignment_operator_from_initializer_list) {
  matrix<double> m(2, 3);

  m = {1, 2, 3};

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 1);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 1);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 3);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3));
}

TEST(matrix, constructor_from_nested_initializer_list) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(matrix, assignment_operator_from_nested_initializer_list) {
  matrix<double> m(2, 4, 10.0);
  m = {{10, 20}, {30, 40}};

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 2);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 2);
  ASSERT_EQ(m.size(), 4);
  ASSERT_EQ(m.capacity(), 8);
  EXPECT_THAT(m, ElementsAre(10, 20, 30, 40));

  m = {{10, 20, 30}, {40, 50, 60}};

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 8);
  EXPECT_THAT(m, ElementsAre(10, 20, 30, 40, 50, 60));
}

TEST(matrix, row_at) {
  matrix<double> m = {{1, 2}, {3, 4}, {5, 6}};
  matrix<double>::row_view first_row = m.row_at(0);

  ASSERT_FALSE(first_row.empty());
  ASSERT_EQ(first_row.num_rows(), 1);
  ASSERT_EQ(first_row.num_cols(), 2);
  ASSERT_EQ(first_row.shape().first, 1);
  ASSERT_EQ(first_row.shape().second, 2);
  ASSERT_EQ(first_row.size(), 2);
  EXPECT_THAT(first_row, ElementsAre(1, 2));

  first_row[0] = 100;

  EXPECT_THAT(m, ElementsAre(100, 2, 3, 4, 5, 6));
}


TEST(matrix, operator_plus_equal_scalar) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};
  m += 10.0;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(11, 12, 13, 14, 15, 16));

  m.row_at(0) += 5.0;

  ASSERT_FALSE(m.empty());
  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);
  ASSERT_EQ(m.size(), 6);
  ASSERT_EQ(m.capacity(), 6);
  EXPECT_THAT(m, ElementsAre(16, 17, 18, 14, 15, 16));

  matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}};
  m1 += 10;

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 2);
  ASSERT_EQ(m1.num_cols(), 3);
  ASSERT_EQ(m1.shape().first, 2);
  ASSERT_EQ(m1.shape().second, 3);
  ASSERT_EQ(m1.size(), 6);
  ASSERT_EQ(m1.capacity(), 6);
  EXPECT_THAT(m1, ElementsAre(11, 12, 13, 14, 15, 16));

  m1.row_at(1) += 5.0;

  ASSERT_FALSE(m1.empty());
  ASSERT_EQ(m1.num_rows(), 2);
  ASSERT_EQ(m1.num_cols(), 3);
  ASSERT_EQ(m1.shape().first, 2);
  ASSERT_EQ(m1.shape().second, 3);
  ASSERT_EQ(m1.size(), 6);
  ASSERT_EQ(m1.capacity(), 6);
  EXPECT_THAT(m1, ElementsAre(11, 12, 13, 19, 20, 21));
}

}  // namespace insight
