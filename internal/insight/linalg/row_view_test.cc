// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
#include "insight/linalg/vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(row_view, of_a_dense_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};

  auto first_row = A.row_at(0);

  EXPECT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row.row_count(), 1);
  EXPECT_EQ(first_row.col_count(), 3);
  EXPECT_EQ(first_row.shape().first, 1);
  EXPECT_EQ(first_row.shape().second, 3);
  EXPECT_THAT(first_row, ElementsAre(0.5, 1.0, 1.5));

  first_row *= 2.0;

  EXPECT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row.row_count(), 1);
  EXPECT_EQ(first_row.col_count(), 3);
  EXPECT_EQ(first_row.shape().first, 1);
  EXPECT_EQ(first_row.shape().second, 3);
  EXPECT_THAT(first_row, ElementsAre(1, 2, 3));
  EXPECT_THAT(A, ElementsAre(1, 2, 3, 2.0, 2.5, 3.0));

  auto second_row = A.row_at(1);

  EXPECT_EQ(second_row.size(), 3);
  EXPECT_EQ(second_row.row_count(), 1);
  EXPECT_EQ(second_row.col_count(), 3);
  EXPECT_EQ(second_row.shape().first, 1);
  EXPECT_EQ(second_row.shape().second, 3);
  EXPECT_THAT(second_row, ElementsAre(2.0, 2.5, 3.0));

  second_row += 0.5;

  EXPECT_EQ(second_row.size(), 3);
  EXPECT_EQ(second_row.row_count(), 1);
  EXPECT_EQ(second_row.col_count(), 3);
  EXPECT_EQ(second_row.shape().first, 1);
  EXPECT_EQ(second_row.shape().second, 3);
  EXPECT_THAT(second_row, ElementsAre(2.5, 3, 3.5));
  EXPECT_THAT(A, ElementsAre(1, 2, 3, 2.5, 3, 3.5));

  matrix<double> x = A.row_at(0);

  EXPECT_EQ(x.size(), 3);
  EXPECT_EQ(x.row_count(), 1);
  EXPECT_EQ(x.col_count(), 3);
  EXPECT_EQ(x.shape().first, 1);
  EXPECT_EQ(x.shape().second, 3);
  EXPECT_THAT(x, ElementsAre(1, 2, 3));
}

TEST(row_view, of_matrix_plus_matrix) {
  matrix<double> A = {{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
  matrix<double> B = {{10, 20, 30}, {40, 50, 60}};

  auto first_row = (A + B).row_at(0);

  EXPECT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row.row_count(), 1);
  EXPECT_EQ(first_row.col_count(), 3);
  EXPECT_EQ(first_row.shape().first, 1);
  EXPECT_EQ(first_row.shape().second, 3);
  EXPECT_THAT(first_row, ElementsAre(10.5, 21, 31.5));

  auto second_row = (A + B).row_at(1);

  EXPECT_EQ(second_row.size(), 3);
  EXPECT_EQ(second_row.row_count(), 1);
  EXPECT_EQ(second_row.col_count(), 3);
  EXPECT_EQ(second_row.shape().first, 1);
  EXPECT_EQ(second_row.shape().second, 3);
  EXPECT_THAT(second_row, ElementsAre(42, 52.5, 63));
}

}  // namespace insight
