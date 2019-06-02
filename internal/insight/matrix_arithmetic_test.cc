// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/matrix.h"
#include "insight/linalg/vector.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(matrix_arithmetic, float_matrix_times_scalar) {
  matrix<double> m1 = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> m = 2.0 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(2, 4, 6, 8, 10, 12));
  EXPECT_THAT(m.row_at(0), ElementsAre(2, 4, 6));
  EXPECT_THAT(m.row_at(1), ElementsAre(8, 10, 12));

  m += -1.0 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 2, 3));
  EXPECT_THAT(m.row_at(1), ElementsAre(4, 5, 6));

  m -= m1 * 3.0;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(-2, -4, -6, -8, -10, -12));
  EXPECT_THAT(m.row_at(0), ElementsAre(-2, -4, -6));
  EXPECT_THAT(m.row_at(1), ElementsAre(-8, -10, -12));

  m /= 2.0 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(-1, -1, -1, -1, -1, -1));
  EXPECT_THAT(m.row_at(0), ElementsAre(-1, -1, -1));
  EXPECT_THAT(m.row_at(1), ElementsAre(-1, -1, -1));

  m *= 2.0 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(-2, -4, -6, -8, -10, -12));
  EXPECT_THAT(m.row_at(0), ElementsAre(-2, -4, -6));
  EXPECT_THAT(m.row_at(1), ElementsAre(-8, -10, -12));
}

TEST(matrix_arithmetic, int_matrix_times_scalar) {
  matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> m = 2.0 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(2, 4, 6, 8, 10, 12));
  EXPECT_THAT(m.row_at(0), ElementsAre(2, 4, 6));
  EXPECT_THAT(m.row_at(1), ElementsAre(8, 10, 12));

  m += -1 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 2, 3));
  EXPECT_THAT(m.row_at(1), ElementsAre(4, 5, 6));

  m -= m1 * 3;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(-2, -4, -6, -8, -10, -12));
  EXPECT_THAT(m.row_at(0), ElementsAre(-2, -4, -6));
  EXPECT_THAT(m.row_at(1), ElementsAre(-8, -10, -12));

  m /= 2 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(-1, -1, -1, -1, -1, -1));
  EXPECT_THAT(m.row_at(0), ElementsAre(-1, -1, -1));
  EXPECT_THAT(m.row_at(1), ElementsAre(-1, -1, -1));

  m *= 2 * m1;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(-2, -4, -6, -8, -10, -12));
  EXPECT_THAT(m.row_at(0), ElementsAre(-2, -4, -6));
  EXPECT_THAT(m.row_at(1), ElementsAre(-8, -10, -12));
}

TEST(matrix_arithmetic, float_matrix_div_scalar) {
  matrix<double> m1 = {{2, 4, 6}, {8, 10, 12}};
  matrix<double> m = m1/2.0;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 2, 3));
  EXPECT_THAT(m.row_at(1), ElementsAre(4, 5, 6));

  m += m1/2.0;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(2, 4, 6, 8, 10, 12));
  EXPECT_THAT(m.row_at(0), ElementsAre(2, 4, 6));
  EXPECT_THAT(m.row_at(1), ElementsAre(8, 10, 12));

  m *= m1/4.0;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 4, 9, 16, 25, 36));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 4, 9));
  EXPECT_THAT(m.row_at(1), ElementsAre(16, 25, 36));

  m /= m1/2.0;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 2, 3));
  EXPECT_THAT(m.row_at(1), ElementsAre(4, 5, 6));
}

TEST(matrix_arithmetic, int_matrix_div_scalar) {
  matrix<int> m1 = {{2, 4, 6}, {8, 10, 12}};
  matrix<int> m = m1/2;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 2, 3));
  EXPECT_THAT(m.row_at(1), ElementsAre(4, 5, 6));

  m += m1/2;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(2, 4, 6, 8, 10, 12));
  EXPECT_THAT(m.row_at(0), ElementsAre(2, 4, 6));
  EXPECT_THAT(m.row_at(1), ElementsAre(8, 10, 12));
}

TEST(matrix_arithmetic, float_matrix_add_matrix) {
  matrix<double> m1 = {{1, 2, 3}, {4, 5, 6}};
  matrix<double> m2 = {{0, 2, 4}, {6, 8, 10}};

  matrix<double> m = m1 + m2;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 4, 7, 10, 13, 16));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 4, 7));
  EXPECT_THAT(m.row_at(1), ElementsAre(10, 13, 16));

  m += m1 + m2;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(2, 8, 14, 20, 26, 32));
  EXPECT_THAT(m.row_at(0), ElementsAre(2, 8, 14));
  EXPECT_THAT(m.row_at(1), ElementsAre(20, 26, 32));

  m -= m1 + m2;

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 4, 7, 10, 13, 16));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 4, 7));
  EXPECT_THAT(m.row_at(1), ElementsAre(10, 13, 16));

  m *= (m1 + m2);

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 16, 49, 100, 169, 256));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 16, 49));
  EXPECT_THAT(m.row_at(1), ElementsAre(100, 169, 256));

  m /= (m1 + m2)*(m1 + m2);

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(1, 1, 1, 1, 1, 1));
  EXPECT_THAT(m.row_at(0), ElementsAre(1, 1, 1));
  EXPECT_THAT(m.row_at(1), ElementsAre(1, 1, 1));
}

TEST(matrix_arithmetic, sqrt) {
  matrix<double> m1 = {{0, 4, 9}, {16, 25, 36}};
  matrix<double> m = sqrt(m1);

  EXPECT_EQ(m.num_rows(), 2);
  EXPECT_EQ(m.num_cols(), 3);
  EXPECT_EQ(m.size(), 6);
  EXPECT_THAT(m, ElementsAre(0, 2, 3, 4, 5, 6));
  EXPECT_THAT(m.row_at(0), ElementsAre(0, 2, 3));
  EXPECT_THAT(m.row_at(1), ElementsAre(4, 5, 6));
}
}  // namespace insight
