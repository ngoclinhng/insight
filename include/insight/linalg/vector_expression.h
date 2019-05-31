// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_LINALG_VECTOR_EXPRESSION_H_
#define INCLUDE_INSIGHT_LINALG_VECTOR_EXPRESSION_H_

#include <functional>
#include <iterator>
#include <type_traits>

#include "glog/logging.h"


namespace insight {

// Base class for all vector expressions.
template<typename Derived>
struct vector_expression {
  const Derived& self() const { return static_cast<const Derived&>(*this); }
};

// Binary expression.

// Element-wise arithmetic between two generic vector expressions.
template<typename E1, typename E2, typename F>
struct vector_binary :
      public vector_expression< vector_binary<E1, E2, F> > {
  // public types.
  using value_type = typename E1::value_type;
  using size_type = typename E1::size_type;
  using difference_type = typename E1::difference_type;
  using const_reference = typename E1::const_reference;
  using reference = const_reference;
  using const_pointer = typename E1::const_pointer;
  using pointer = const_pointer;
  using shape_type = typename E1::shape_type;

  const E1& e1;
  const E2& e2;
  const F& f;

  vector_binary(const E1& e1, const E2& e2, const F& f)
      : e1(e1), e2(e2), f(f) {
    CHECK_EQ(e1.size(), e2.size());
  }

  inline size_type num_rows() const { return e1.num_rows(); }
  inline size_type num_cols() const { return e1.num_cols(); }
  inline shape_type shape() const { return e1.shape(); }
  inline size_type size() const { return e1.size(); }

  // Iterator.

  class const_iterator;
  using iterator = const_iterator;

  class const_iterator {
   private:
    using const_subiterator1_type = typename E1::const_iterator;
    using const_subiterator2_type = typename E2::const_iterator;

   public:
    // public types.
    using value_type = typename vector_binary::value_type;
    using difference_type = typename vector_binary::difference_type;
    using pointer = typename vector_binary::const_pointer;
    using reference = typename vector_binary::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : index_(), expr_(), it1_(), it2_() {}

    const_iterator(const vector_binary& expr, size_type index)
        : expr_(expr), index_(index),
          it1_(expr.e1.begin()),
          it2_(expr.e2.begin()) {}

    // Copy constructor.
    const_iterator(const const_iterator& it)
        : expr_(it.expr_),
          index_(it.index_),
          it1_(it.it1_),
          it2_(it.it2_) {
    }

    // Assignment operator.
    const_iterator& operator=(const const_iterator& it) {
      if (this == &it) { return *this; }
      expr_ = it.expr_;
      index_ = it.index_;
      it1_ = it.it1_;
      it2_ = it.it2_;
      return *this;
    }

    // Dereference.
    inline value_type operator*() const {
      return expr_.f(*it1_, *it2_);
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) {
      return (index_ == it.index_);
    }

    inline bool operator!=(const const_iterator& it) {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline const_iterator& operator++() {
      ++index_;
      ++it1_;
      ++it2_;
      return *this;
    }

    // Postfix increment it++.
    inline const_iterator operator++(int) {
      const_iterator temp(*this);
      ++(*this);
      return temp;
    }

   private:
    const vector_binary& expr_;
    size_type index_;
    const_subiterator1_type it1_;
    const_subiterator2_type it2_;
  };

  inline const_iterator begin() const { return const_iterator(*this, 0); }
  inline const_iterator cbegin() const { return begin(); }
  inline const_iterator end() const { return const_iterator(*this, size()); }
  inline const_iterator cend() const  {return end(); }
};

// Element-wise arithmetic between a generic vector expression
// and a scalar.
template<typename E, typename F>
struct vector_binary<E, typename E::value_type, F>
    : public vector_expression<vector_binary<E, typename E::value_type, F> > {  // NOLINT
  // public types.
  using value_type = typename E::value_type;
  using size_type = typename E::size_type;
  using difference_type = typename E::difference_type;
  using const_reference = typename E::const_reference;
  using reference = const_reference;
  using const_pointer = typename E::const_pointer;
  using pointer = const_pointer;
  using shape_type = typename E::shape_type;


  const E& e;
  const value_type scalar;
  const F& f;

  vector_binary(const E& e, value_type scalar, const F& f)
      : e(e), scalar(scalar), f(f) {}

  inline size_type num_rows() const { return e.num_rows(); }
  inline size_type num_cols() const { return e.num_cols(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  // Iterator.

  class const_iterator;
  using iterator = const_iterator;

  class const_iterator {
   private:
    using const_subiterator_type = typename E::const_iterator;

   public:
    // public types.
    using value_type = typename vector_binary::value_type;
    using difference_type = typename vector_binary::difference_type;
    using pointer = typename vector_binary::const_pointer;
    using reference = typename vector_binary::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : index_(), expr_(), it_() {}

    const_iterator(const vector_binary& expr, size_type index)
        : expr_(expr), index_(index), it_(expr.e.begin()) {}

    // Copy constructor.
    const_iterator(const const_iterator& it)
        : expr_(it.expr_), index_(it.index_), it_(it.it_) {}

    // Assignment operator.
    const_iterator& operator=(const const_iterator& it) {
      if (this == &it) { return *this; }
      expr_ = it.expr_;
      index_ = it.index_;
      it_ = it.it_;
      return *this;
    }

    // Dereference.
    inline value_type operator*() const {
      return expr_.f(*it_, expr_.scalar);
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) {
      return (index_ == it.index_);
    }

    inline bool operator!=(const const_iterator& it) {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline const_iterator& operator++() {
      ++index_;
      ++it_;
      return *this;
    }

    // Postfix increment it++.
    inline const_iterator operator++(int) {
      const_iterator temp(*this);
      ++(*this);
      return temp;
    }

   private:
    const vector_binary& expr_;
    size_type index_;
    const_subiterator_type it_;
  };

  inline const_iterator begin() const { return const_iterator(*this, 0); }
  inline const_iterator cbegin() const { return begin(); }
  inline const_iterator end() const { return const_iterator(*this, size()); }
  inline const_iterator cend() const  {return end(); }
};

// Element-wise arithmetic between a scalar and a generic vector
// expression.
template<typename E, typename F>
struct vector_binary<typename E::value_type, E, F>
    : public vector_expression<vector_binary<typename E::value_type, E, F> > {  // NOLINT
  // Public types.
  using value_type = typename E::value_type;
  using size_type = typename E::size_type;
  using difference_type = typename E::difference_type;
  using const_reference = typename E::const_reference;
  using reference = const_reference;
  using const_pointer = typename E::const_pointer;
  using pointer = const_pointer;
  using shape_type = typename E::shape_type;

  const value_type scalar;
  const E& e;
  const F& f;

  vector_binary(value_type scalar, const E& e, const F& f)
      : scalar(scalar), e(e), f(f) {}

  inline size_type num_rows() const { return e.num_rows(); }
  inline size_type num_cols() const { return e.num_cols(); }
  inline shape_type shape() const { return e.shape(); }
  inline size_type size() const { return e.size(); }

  inline value_type operator[](size_type i) const {
    return f(scalar, e[i]);
  }

  // Iterator.

  class const_iterator;
  using iterator = const_iterator;

  class const_iterator {
   private:
    using const_subiterator_type = typename E::const_iterator;

   public:
    // public types.
    using value_type = typename vector_binary::value_type;
    using difference_type = typename vector_binary::difference_type;
    using pointer = typename vector_binary::const_pointer;
    using reference = typename vector_binary::const_reference;
    using iterator_category = std::input_iterator_tag;

    const_iterator() : index_(), expr_(), it_() {}

    const_iterator(const vector_binary& expr, size_type index)
        : expr_(expr), index_(index),
          it_(expr.e.begin()) {}

    // Copy constructor.
    const_iterator(const const_iterator& it)
        : expr_(it.expr_), index_(it.index_), it_(it.it_) { }

    // Assignment operator.
    const_iterator& operator=(const const_iterator& it) {
      if (this == &it) { return *this; }
      expr_ = it.expr_;
      index_ = it.index_;
      it_ = it.it_;
      return *this;
    }

    // Dereference.
    inline value_type operator*() const {
      return expr_.f(expr_.scalar, *it_);
    }

    // Comparison.

    inline bool operator==(const const_iterator& it) {
      return (index_ == it.index_);
    }

    inline bool operator!=(const const_iterator& it) {
      return !(*this == it);
    }

    // Prefix increment ++it.
    inline const_iterator& operator++() {
      ++index_;
      ++it_;
      return *this;
    }

    // Postfix increment it++.
    inline const_iterator operator++(int) {
      const_iterator temp(*this);
      ++(*this);
      return temp;
    }

   private:
    const vector_binary& expr_;
    size_type index_;
    const_subiterator_type it_;
  };

  inline const_iterator begin() const { return const_iterator(*this, 0); }
  inline const_iterator cbegin() const { return begin(); }
  inline const_iterator end() const { return const_iterator(*this, size()); }
  inline const_iterator cend() const  {return end(); }
};

// overload operators.

// Addition between two generic vector expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for addition
// between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  vector_binary<L, R, std::plus<typename L::value_type> >
  >::type
operator+(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return vector_binary<
    L, R,
    std::plus<typename L::value_type>
    >(e1.self(), e2.self(), std::plus<typename L::value_type>());
}

// addition between a generic vector expression and a scalar.
template<typename E>
inline
vector_binary<E, typename E::value_type,
                  std::plus<typename E::value_type> >
operator+(const vector_expression<E>& e, typename E::value_type scalar) {
  return vector_binary<
    E,
    typename E::value_type,
    std::plus<typename E::value_type>
    >(e.self(), scalar, std::plus<typename E::value_type>());
}

// addition between a scalar and a generic vector expression.
template<typename E>
inline
vector_binary<typename E::value_type, E,
                  std::plus<typename E::value_type> >
operator+(typename E::value_type scalar, const vector_expression<E>& e) {
  return vector_binary<
    typename E::value_type,
    E,
    std::plus<typename E::value_type>
    >(scalar, e.self(), std::plus<typename E::value_type>());
}

// Substraction between two generic vector expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for substraction
// between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  vector_binary<L, R, std::minus<typename L::value_type> >
  >::type
operator-(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return vector_binary<
    L, R,
    std::minus<typename L::value_type>
    >(e1.self(), e2.self(), std::minus<typename L::value_type>());
}

// Substraction between a generic vector expression and a scalar.
template<typename E>
inline
vector_binary<E, typename E::value_type,
                  std::minus<typename E::value_type> >
operator-(const vector_expression<E>& e, typename E::value_type scalar) {
  return vector_binary<
    E,
    typename E::value_type,
    std::minus<typename E::value_type>
    >(e.self(), scalar, std::minus<typename E::value_type>());
}

// Substraction between a scalar and a generic vector expression.
template<typename E>
inline
vector_binary<typename E::value_type, E,
                  std::minus<typename E::value_type> >
operator-(typename E::value_type scalar, const vector_expression<E>& e) {
  return vector_binary<
    typename E::value_type,
    E,
    std::minus<typename E::value_type>
    >(scalar, e.self(), std::minus<typename E::value_type>());
}

// Element-wise multiplication between two generic vector
// expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// multiplication between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  vector_binary<L, R, std::multiplies<typename L::value_type> >
  >::type
operator*(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return vector_binary<
    L, R,
    std::multiplies<typename L::value_type>
    >(e1.self(), e2.self(), std::multiplies<typename L::value_type>());
}

// Multiplication between a generic vector expression and a scalar.
template<typename E>
inline
vector_binary<E, typename E::value_type,
            std::multiplies<typename E::value_type> >
operator*(const vector_expression<E>& e, typename E::value_type scalar) {
  return vector_binary<
    E,
    typename E::value_type,
    std::multiplies<typename E::value_type>
    >(e.self(), scalar, std::multiplies<typename E::value_type>());
}

// Multiplication between a scalar and a generic vector expression.
template<typename E>
inline
vector_binary<typename E::value_type, E,
            std::multiplies<typename E::value_type> >
operator*(typename E::value_type scalar, const vector_expression<E>& e) {
  return vector_binary<
    typename E::value_type,
    E,
    std::multiplies<typename E::value_type>
    >(scalar, e.self(), std::multiplies<typename E::value_type>());
}

// Element-wise division between two generic vector expressions.
template<typename L, typename R>
inline
// We need to make sure that whose ever L and R are they must have the same
// element type. So that, for example no viable overload for element-wise
// division between an integer vector and a floating-point vector.
typename
std::enable_if<
  std::is_same<typename L::value_type, typename R::value_type>::value,
  vector_binary<L, R, std::divides<typename L::value_type> >
  >::type
operator/(const vector_expression<L>& e1, const vector_expression<R>& e2) {
  return vector_binary<
    L, R,
    std::divides<typename L::value_type>
    >(e1.self(), e2.self(), std::divides<typename L::value_type>());
}

// Element-wise division between a generic vector expression and
// a scalar.
template<typename E>
inline
vector_binary<E, typename E::value_type,
                  std::divides<typename E::value_type> >
operator/(const vector_expression<E>& e, typename E::value_type scalar) {
  return vector_binary<
    E,
    typename E::value_type,
    std::divides<typename E::value_type>
    >(e.self(), scalar, std::divides<typename E::value_type>());
}

// Element-wise division between a scalar and a generic vector
// expression.
template<typename E>
inline
vector_binary<typename E::value_type, E,
                  std::divides<typename E::value_type> >
operator/(typename E::value_type scalar, const vector_expression<E>& e) {
  return vector_binary<
    typename E::value_type,
    E,
    std::divides<typename E::value_type>
    >(scalar, e.self(), std::divides<typename E::value_type>());
}

}  // namespace insight
#endif  // INCLUDE_INSIGHT_LINALG_VECTOR_EXPRESSION_H_
