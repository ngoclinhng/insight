// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/detail/transpose_expression.h"
#include "insight/linalg/matrix.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {
namespace linalg_detail {

TEST(transpose_iterator, increment_by_one) {
  matrix<int> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  auto t_begin = make_transpose_iterator(A.begin(), 0,
                                         A.row_count(),
                                         A.col_count());
  auto t_end = make_transpose_iterator(A.begin(),
                                       A.size(),
                                       A.row_count(),
                                       A.col_count());
  EXPECT_EQ(t_end - t_begin, 12);

  EXPECT_EQ(*t_begin, 1);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 11);
  EXPECT_EQ(*t_begin, 5);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 10);
  EXPECT_EQ(*t_begin, 9);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 9);
  EXPECT_EQ(*t_begin, 2);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 8);
  EXPECT_EQ(*t_begin, 6);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 7);
  EXPECT_EQ(*t_begin, 10);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 6);
  EXPECT_EQ(*t_begin, 3);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 5);
  EXPECT_EQ(*t_begin, 7);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 4);
  EXPECT_EQ(*t_begin, 11);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 3);
  EXPECT_EQ(*t_begin, 4);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 2);
  EXPECT_EQ(*t_begin, 8);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 1);
  EXPECT_EQ(*t_begin, 12);
  ++t_begin;
  EXPECT_EQ(t_end - t_begin, 0);
  EXPECT_TRUE(t_begin == t_end);
}

TEST(transpose_iterator, decrement_by_one) {
  matrix<int> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  auto t_begin = make_transpose_iterator(A.begin(), 0,
                                         A.row_count(),
                                         A.col_count());
  auto t_end = make_transpose_iterator(A.begin(),
                                       A.size(),
                                       A.row_count(),
                                       A.col_count());
  EXPECT_EQ(t_end - t_begin, 12);

  --t_end;
  EXPECT_EQ(t_end - t_begin, 11);
  EXPECT_EQ(*t_end, 12);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 10);
  EXPECT_EQ(*t_end, 8);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 9);
  EXPECT_EQ(*t_end, 4);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 8);
  EXPECT_EQ(*t_end, 11);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 7);
  EXPECT_EQ(*t_end, 7);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 6);
  EXPECT_EQ(*t_end, 3);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 5);
  EXPECT_EQ(*t_end, 10);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 4);
  EXPECT_EQ(*t_end, 6);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 3);
  EXPECT_EQ(*t_end, 2);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 2);
  EXPECT_EQ(*t_end, 9);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 1);
  EXPECT_EQ(*t_end, 5);
  --t_end;
  EXPECT_EQ(t_end - t_begin, 0);
  EXPECT_EQ(*t_end, 1);
  EXPECT_TRUE(t_end == t_begin);
}

TEST(transpose_iterator, increment_by_n) {
  matrix<int> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  auto t_begin = make_transpose_iterator(A.begin(), 0,
                                         A.row_count(),
                                         A.col_count());
  auto t_end = make_transpose_iterator(A.begin(),
                                       A.size(),
                                       A.row_count(),
                                       A.col_count());

  t_begin += 2;
  EXPECT_EQ(*t_begin, 9);
  t_begin += 2;
  EXPECT_EQ(*t_begin, 6);
  t_begin += 4;
  EXPECT_EQ(*t_begin, 11);
  t_begin += 4;
  EXPECT_TRUE(t_begin == t_end);

  t_begin -= 3;
  EXPECT_EQ(*t_begin, 4);
  t_begin -= 4;
  EXPECT_EQ(*t_begin, 10);
}

TEST(transpose_iterator, decrement_by_n) {
  matrix<int> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  auto t_begin = make_transpose_iterator(A.begin(), 0,
                                         A.row_count(),
                                         A.col_count());
  auto t_end = make_transpose_iterator(A.begin(),
                                       A.size(),
                                       A.row_count(),
                                       A.col_count());

  // t_begin | 1  5  9 | 2  6  10 | 3  7  11 | 4  8  12 | t_end
  t_end -= 2;
  EXPECT_EQ(*t_end, 8);
  t_end -= 6;
  EXPECT_EQ(*t_end, 6);
  t_end -= 4;
  EXPECT_EQ(*t_end, 1);
  EXPECT_TRUE(t_end == t_begin);
}

TEST(transpose_iterator, matrix_expression) {
  matrix<int> A = {{1, 2, 3}, {4, 5, 6}};
  matrix<int> B = {{10, 20, 30}, {40, 50, 60}};

  auto e = A + B;

  auto t_begin = make_transpose_iterator(e.begin(), 0,
                                         e.row_count(),
                                         e.col_count());
  auto t_end = make_transpose_iterator(e.begin(),
                                       e.size(),
                                       e.row_count(),
                                       e.col_count());

  // t_begin | 11  44 | 22  55 | 33  66 | t_end
  EXPECT_EQ(*t_begin, 11);
  ++t_begin;
  EXPECT_EQ(*t_begin, 44);
  ++t_begin;
  EXPECT_EQ(*t_begin, 22);
  ++t_begin;
  EXPECT_EQ(*t_begin, 55);
  ++t_begin;
  EXPECT_EQ(*t_begin, 33);
  ++t_begin;
  EXPECT_EQ(*t_begin, 66);
  ++t_begin;
  EXPECT_TRUE(t_begin == t_end);
}
}  // namespace linalg_detail
}  // namespace insight
