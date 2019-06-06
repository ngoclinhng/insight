// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
#include "insight/linalg/functions.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(row_view, of_a_dense_float_matrix) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};

  auto first_row = A.row_at(0);

  EXPECT_EQ(first_row.num_rows(), 1);
  EXPECT_EQ(first_row.num_cols(), 3);
  EXPECT_EQ(first_row.size(), 3);
  EXPECT_THAT(first_row, ElementsAre(1, 2, 3));

  first_row *= 10.0;

  EXPECT_EQ(A.num_rows(), 2);
  EXPECT_EQ(A.num_cols(), 3);
  EXPECT_EQ(A.size(), 6);
  EXPECT_THAT(A, ElementsAre(10, 20, 30, 4, 5, 6));

  A *= 2.0;
  auto second_row = A.row_at(1);

  EXPECT_EQ(second_row.num_rows(), 1);
  EXPECT_EQ(second_row.num_cols(), 3);
  EXPECT_EQ(second_row.size(), 3);
  EXPECT_THAT(second_row, ElementsAre(8, 10, 12));
  EXPECT_THAT(first_row, ElementsAre(20, 40, 60));
}

TEST(row_view, of_a_binary_matrix_expression) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{10, 20, 30}, {40, 50, 60}};

  auto e = A + B;

  EXPECT_EQ(e.num_rows(), 2);
  EXPECT_EQ(e.num_cols(), 3);
  EXPECT_EQ(e.size(), 6);
  EXPECT_THAT(e, ElementsAre(11, 22, 33, 44, 55, 66));
  EXPECT_THAT(e.row_at(0), ElementsAre(11, 22, 33));
  EXPECT_THAT(e.row_at(1), ElementsAre(44, 55, 66));
}

TEST(row_view, of_a_unary_matrix_expression) {
  matrix<double> A = {{0, 4, 9}, {16, 25, 36}};
  auto e = sqrt(A);

  EXPECT_EQ(e.num_rows(), 2);
  EXPECT_EQ(e.num_cols(), 3);
  EXPECT_EQ(e.size(), 6);
  EXPECT_THAT(e, ElementsAre(0, 2, 3, 4, 5, 6));
  EXPECT_THAT(e.row_at(0), ElementsAre(0, 2, 3));
  EXPECT_THAT(e.row_at(1), ElementsAre(4, 5, 6));
}

TEST(row_view, of_a_transposed_matrix) {
  matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  auto e = A.t();

  EXPECT_EQ(e.num_rows(), 3);
  EXPECT_EQ(e.num_cols(), 2);
  EXPECT_EQ(e.size(), 6);
  EXPECT_THAT(e, ElementsAre(1, 4, 2, 5, 3, 6));
  EXPECT_THAT(e.row_at(0), ElementsAre(1, 4));
  EXPECT_THAT(e.row_at(1), ElementsAre(2, 5));
  EXPECT_THAT(e.row_at(2), ElementsAre(3, 6));
}
}  // namespace insight
