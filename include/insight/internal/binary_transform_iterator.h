// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_BINARY_TRANSFORM_ITERATOR_H_
#define INCLUDE_INSIGHT_INTERNAL_BINARY_TRANSFORM_ITERATOR_H_

#include <iterator>
#include <type_traits>

namespace insight {
namespace internal {

template<typename Iter1, typename Iter2, typename Functor>
class binary_transform_iterator {
 private:
  Iter1 it1_;
  Iter2 it2_;
  Functor f_;

  using iter1_traits = std::iterator_traits<Iter1>;
  using iter2_traits = std::iterator_traits<Iter2>;
  using reference1_ = typename iter1_traits::reference;
  using reference2_ = typename iter2_traits::reference;

 public:
  static_assert(std::is_same<typename iter1_traits::difference_type,
                typename iter2_traits::difference_type>::value,
                "Iter1 and Iter2 must have the same difference_type");

  static_assert(std::is_same<typename iter1_traits::iterator_category,
                typename iter2_traits::iterator_category>::value,
                "Iter1 and Iter2 must have the same iterator_category");

  using iterator1_type = Iter1;
  using iterator2_type = Iter2;

  using iterator_category = typename iter1_traits::iterator_category;
  // TODO(Linh): This is deprecated in C++17, and removed in C++20.
  using value_type = typename std::result_of<Functor(reference1_, reference2_)>::type;  // NOLINT
  using difference_type = typename iter1_traits::difference_type;
  // TODO(Linh): void or what?
  using pointer = void;
  // TODO(Linh): why not value_type& or even value_type?
  using reference = value_type;

  binary_transform_iterator() : it1_(), it2_(), f_() {}

  binary_transform_iterator(const Iter1& it1,
                            const Iter2& it2,
                            const Functor& f)
      : it1_(it1), it2_(it2), f_(f) {}

  template<typename U1, typename U2>
  binary_transform_iterator(const binary_transform_iterator<U1, U2, Functor>& src)  // NOLINT
      : it1_(src.base1()),
        it2_(src.base2()),
        f_(src.functor()) {}

  Iter1 base1() const { return it1_; }
  Iter2 base2() const { return it2_; }
  Functor functor() const { return f_; }

  reference operator*() const {
    return static_cast<reference>(f_(*it1_, *it2_));
  }

  // TODO(Linh): neccessary?
  // pointer  operator->() const { return it_;}

  binary_transform_iterator& operator++() {
    ++it1_;
    ++it2_;
    return *this;
  }

  binary_transform_iterator  operator++(int) {
    binary_transform_iterator tmp(*this);
    ++it1_;
    ++it2_;
    return tmp;
  }

  binary_transform_iterator& operator--() {
    --it1_;
    --it2_;
    return *this;
  }

  binary_transform_iterator  operator--(int) {
    binary_transform_iterator tmp(*this);
    --it1_;
    --it2_;
    return tmp;
  }

  binary_transform_iterator  operator+ (difference_type n) const {
    return binary_transform_iterator(it1_ + n, it2_ + n, f_);
  }

  binary_transform_iterator& operator+=(difference_type n) {
    it1_ += n;
    it2_ += n;
    return *this;
  }

  binary_transform_iterator  operator- (difference_type n) const {
    return binary_transform_iterator(it1_ - n, it2_ - n, f_);
  }

  binary_transform_iterator& operator-=(difference_type n) {
    it1_ -= n;
    it2_ -= n;
    return *this;
  }

  reference operator[](difference_type n) const {
    return static_cast<reference>(f_(it1_[n], it2_[n]));
  }
};


template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
bool
operator==(const binary_transform_iterator<T1, T2, Functor>& x,
           const binary_transform_iterator<U1, U2, Functor>& y) {
  return (x.base1() == y.base1()) && (x.base2() == y.base2());
}

template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
bool
operator!=(const binary_transform_iterator<T1, T2, Functor>& x,
           const binary_transform_iterator<U1, U2, Functor>& y) {
  return (x.base1() != y.base1()) || (x.base2() != y.base2());
}


template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
bool
operator<(const binary_transform_iterator<T1, T2, Functor>& x,
           const binary_transform_iterator<U1, U2, Functor>& y) {
  return (x.base1() < y.base1()) && (x.base2() < y.base2());
}

template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
bool
operator<=(const binary_transform_iterator<T1, T2, Functor>& x,
          const binary_transform_iterator<U1, U2, Functor>& y) {
  return (x.base1() <= y.base1()) && (x.base2() <= y.base2());
}

template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
bool
operator>(const binary_transform_iterator<T1, T2, Functor>& x,
          const binary_transform_iterator<U1, U2, Functor>& y) {
  return (x.base1() > y.base1()) && (x.base2() > y.base2());
}

template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
bool
operator>=(const binary_transform_iterator<T1, T2, Functor>& x,
           const binary_transform_iterator<U1, U2, Functor>& y) {
  return (x.base1() >= y.base1()) && (x.base2() >= y.base2());
}

template<typename T1, typename T2, typename U1, typename U2, typename Functor>
inline
auto
operator-(const binary_transform_iterator<T1, T2, Functor>& x,
          const binary_transform_iterator<U1, U2, Functor>& y)
    -> decltype(x.base1() - y.base1()) {
  // TODO(Linh): Need to make sure that x.base2() - y.base2() is equal
  // to x.base1() - y.base1().
  return x.base1() - y.base1();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
binary_transform_iterator<Iter1, Iter2, Functor>
operator+(typename
          binary_transform_iterator<Iter1, Iter2, Functor>::difference_type n,
          const binary_transform_iterator<Iter1, Iter2, Functor>& x) {
  return binary_transform_iterator<Iter1, Iter2, Functor>(x.base1() + n,
                                                          x.base2() + n,
                                                          x.functor());
}

template<typename Iter1, typename Iter2, typename Functor>
inline
binary_transform_iterator<Iter1, Iter2, Functor>
make_binary_transform_iterator(const Iter1& it1, const Iter2& it2,
                               const Functor& f) {
  return binary_transform_iterator<Iter1, Iter2, Functor>(it1, it2, f);
}

}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_BINARY_TRANSFORM_ITERATOR_H_
