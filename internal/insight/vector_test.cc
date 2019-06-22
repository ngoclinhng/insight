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

TEST(vector, default_constructor) {
  vector<double> vec;

  EXPECT_TRUE(vec.empty());
  EXPECT_EQ(vec.size(), 0);
}

TEST(vector, default_constructed_from_given_size) {
  vector<double> v(5);

  EXPECT_FALSE(v.empty());
  EXPECT_EQ(v.size(), 5);
  EXPECT_THAT(v, ElementsAre(0, 0, 0, 0, 0));
}

TEST(vector, copy_constructed_from_given_value) {
  vector<double> dvec(4, 0.5);

  EXPECT_FALSE(dvec.empty());
  EXPECT_EQ(dvec.size(), 4);
  EXPECT_THAT(dvec, ElementsAre(0.5, 0.5, 0.5, 0.5));

  vector<int> ivec(5, 10);

  EXPECT_FALSE(ivec.empty());
  EXPECT_EQ(ivec.size(), 5);
  EXPECT_THAT(ivec, ElementsAre(10, 10, 10, 10, 10));
}

TEST(vector, construct_from_range) {
  double ddata[6] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  vector<double> dvec(ddata, ddata + 6);

  EXPECT_FALSE(dvec.empty());
  EXPECT_EQ(dvec.size(), 6);
  EXPECT_THAT(dvec, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));

  int idata[4] = {10, 20, 30, 40};
  vector<int> ivec(idata, idata + 4);

  EXPECT_FALSE(ivec.empty());
  EXPECT_EQ(ivec.size(), 4);
  EXPECT_THAT(ivec, ElementsAre(10, 20, 30, 40));
}

TEST(vector, construct_from_initializer_list) {
  vector<double> dvec = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

  EXPECT_FALSE(dvec.empty());
  EXPECT_EQ(dvec.size(), 6);
  EXPECT_THAT(dvec, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));

  vector<int> ivec = {10, 20, 30, 40};

  EXPECT_FALSE(ivec.empty());
  EXPECT_EQ(ivec.size(), 4);
  EXPECT_THAT(ivec, ElementsAre(10, 20, 30, 40));
}

TEST(vector, copy_constructor) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  vector<double> y = {-0.5, 1.0, -1.5};

  // Memory allocated
  vector<double> z = x;

  EXPECT_FALSE(z.empty());
  EXPECT_EQ(z.size(), 6);
  EXPECT_THAT(z, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));

  // No memory allocated.
  z = y;

  EXPECT_FALSE(z.empty());
  EXPECT_EQ(z.size(), 3);
  EXPECT_THAT(z, ElementsAre(-0.5, 1.0, -1.5));
}

TEST(vector, assignment_operator) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  vector<double> y = {-0.5, 1.0, -1.5};
  vector<double> z;

  // Memory allocated.
  z = x;

  EXPECT_FALSE(z.empty());
  EXPECT_EQ(z.size(), 6);
  EXPECT_THAT(z, ElementsAre(0.5, 1.0, 1.5, 2.0, 2.5, 3.0));

  // No memory allocated.
  z = y;

  EXPECT_FALSE(z.empty());
  EXPECT_EQ(z.size(), 3);
  EXPECT_THAT(z, ElementsAre(-0.5, 1.0, -1.5));
}

TEST(vector, add_scalar) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  x += 10.5;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(11, 11.5, 12, 12.5, 13, 13.5));

  vector<int> y = {10, 20, 30};
  y += 5;

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(15, 25, 35));
}

TEST(vector, sub_scalar) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  x -= 0.5;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(0, 0.5, 1.0, 1.5, 2.0, 2.5));

  vector<int> y = {10, 20, 30};
  y -= 5;

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(5, 15, 25));
}

TEST(vector, mul_scalar) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  x *= 2.0;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(1, 2, 3, 4, 5, 6));

  vector<int> y = {10, 20, 30};
  y *= 2;

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(20, 40, 60));
}

TEST(vector, div_scalar) {
  vector<double> x = {3, 9, 12, 21, 24, -9};
  x /= 3.0;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(1, 3, 4, 7, 8, -3));

  vector<int> y = {10, -20, 30};
  y /= 10;

  EXPECT_FALSE(y.empty());
  EXPECT_EQ(y.size(), 3);
  EXPECT_THAT(y, ElementsAre(1, -2, 3));
}

TEST(vector, add_vector) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  vector<double> y = {-1.5, 0, 2.5, 3, 4, 8};

  x += y;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(-1, 1, 4, 5, 6.5, 11));

  vector<int> a = {1, 2, 3, 4};
  vector<int> b = {10, 20, 30, 40};

  a += b;

  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a.size(), 4);
  EXPECT_THAT(a, ElementsAre(11, 22, 33, 44));
}

TEST(vector, sub_vector) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  vector<double> y = {-1.5, 0, 2.5, 3, 4, 8};

  x -= y;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(2, 1, -1, -1, -1.5, -5));

  vector<int> a = {1, 2, 3, 4};
  vector<int> b = {10, 20, 30, 40};

  a -= b;

  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a.size(), 4);
  EXPECT_THAT(a, ElementsAre(-9, -18, -27, -36));
}

TEST(vector, mul_vector) {
  vector<double> x = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  vector<double> y = {1.0, 0.2, 2.0, -3.0, 1.0, 5.0};

  x *= y;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(0.5, 0.2, 3.0, -6, 2.5, 15));

  vector<int> a = {1, 2, 3, 4, 5};
  vector<int> b = {-5, 6, 7, 2, -3};

  a *= b;

  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a.size(), 5);
  EXPECT_THAT(a, ElementsAre(-5, 12, 21, 8, -15));
}

TEST(vector, div_vector) {
  vector<double> x = {10, 20, 30, 40, 50, 60};
  vector<double> y = {1, 2, 3, 4, 5, 6};

  x /= y;

  EXPECT_FALSE(x.empty());
  EXPECT_EQ(x.size(), 6);
  EXPECT_THAT(x, ElementsAre(10, 10, 10, 10, 10, 10));

  vector<int> a = {15, 26, 28};
  vector<int> b = {3, 2, 7};

  a /= b;

  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a.size(), 3);
  EXPECT_THAT(a, ElementsAre(5, 13, 4));
}

TEST(vector, nrm2) {
  vector<double> x = {3, 2, 2, 2, 2};
  EXPECT_THAT(x.nrm2(), DoubleEq(5.0));

  vector<int> a = {3, 4, 0};
  EXPECT_EQ(a.nrm2(), 5);
}

TEST(vector, dot) {
  vector<double> x = {1.5, 2, -0.5, 4, 3, 9};
  vector<double> y = {1.0, -2.5, 1.5, 4, 6, 8};

  EXPECT_THAT(x.dot(y), DoubleEq(101.75));
  EXPECT_THAT(y.dot(x), DoubleEq(101.75));

  vector<int> a = {1, 2, 3};
  vector<int> b = {10, 20, 30};

  EXPECT_EQ(a.dot(b), 140);
  EXPECT_EQ(b.dot(a), 140);
}

}  // namespace insight
