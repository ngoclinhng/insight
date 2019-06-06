// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(col_view, of_a_float_dense_matrix) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};

  auto first_col = m.col_at(0);

  EXPECT_EQ(first_col.num_rows(), 2);
  EXPECT_EQ(first_col.num_cols(), 1);
  EXPECT_EQ(first_col.size(), 2);
  EXPECT_THAT(first_col, ElementsAre(1, 4));

  auto second_col = m.col_at(1);

  EXPECT_EQ(second_col.num_rows(), 2);
  EXPECT_EQ(second_col.num_cols(), 1);
  EXPECT_EQ(second_col.size(), 2);
  EXPECT_THAT(second_col, ElementsAre(2, 5));

  auto third_col = m.col_at(2);

  EXPECT_EQ(third_col.num_rows(), 2);
  EXPECT_EQ(third_col.num_cols(), 1);
  EXPECT_EQ(third_col.size(), 2);
  EXPECT_THAT(third_col, ElementsAre(3, 6));

  m.col_at(0) *= 2.0;

  EXPECT_THAT(m, ElementsAre(2, 2, 3, 8, 5, 6));
  EXPECT_THAT(m.col_at(0), ElementsAre(2, 8));
  EXPECT_THAT(m.col_at(1), ElementsAre(2, 5));
  EXPECT_THAT(m.col_at(2), ElementsAre(3, 6));
}

TEST(col_view, of_a_dense_matrix_expression) {
  matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
  matrix<double> B = {{10, 20}, {30, 40}, {50, 60}};

  auto e = A + B;

  EXPECT_THAT(e.num_rows(), 3);
  EXPECT_THAT(e.num_cols(), 2);
  EXPECT_THAT(e.size(), 6);
  EXPECT_THAT(e.col_at(0), ElementsAre(11, 33, 55));
  EXPECT_THAT(e.col_at(1), ElementsAre(22, 44, 66));
}

TEST(col_view, of_a_dense_matrix_transpose) {
  matrix<double> m = {{10, 20, 30}, {40, 50, 60}};
  auto e = m.t();

  EXPECT_THAT(e.num_rows(), 3);
  EXPECT_THAT(e.num_cols(), 2);
  EXPECT_THAT(e.size(), 6);
  EXPECT_THAT(e.col_at(0), ElementsAre(10, 20, 30));
  EXPECT_THAT(e.col_at(1), ElementsAre(40, 50, 60));
}
}  // namespace insight
