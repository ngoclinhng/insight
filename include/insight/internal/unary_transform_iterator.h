// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_UNARY_TRANSFORM_ITERATOR_H_
#define INCLUDE_INSIGHT_INTERNAL_UNARY_TRANSFORM_ITERATOR_H_

#include <iterator>
#include <type_traits>

namespace insight {
namespace internal {

template<typename Iter, typename Functor>
class unary_transform_iterator {
 private:
  Iter it_;
  Functor f_;

  using iter_traits = std::iterator_traits<Iter>;
  using reference_ = typename iter_traits::reference;

 public:
  using iterator_type = Iter;
  using iterator_category = typename iter_traits::iterator_category;
  // TODO(Linh): This is deprecated in C++17, and removed in C++20.
  using value_type = typename std::result_of<Functor(reference_)>::type;
  using difference_type = typename iter_traits::difference_type;
  // TODO(Linh): void or what?
  using pointer = void;
  // TODO(Linh): why not value_type& or even value_type?
  using reference = value_type;

  // TODO(Linh): Need to make sure that Functor is callable and the return
  // type when we call f_(*it_) should be consistent with value_type/reference.

  unary_transform_iterator() : it_(), f_() {}

  unary_transform_iterator(const Iter& it, const Functor& f)
      : it_(it), f_(f) {}

  template<typename U>
  unary_transform_iterator(const unary_transform_iterator<U, Functor>& src)
      : it_(src.base()), f_(src.functor()) {}

  Iter base() const { return it_; }
  Functor functor() const { return f_; }

  reference operator*() const { return static_cast<reference>(f_(*it_)); }

  // TODO(Linh): neccessary?
  // pointer  operator->() const { return it_;}

  unary_transform_iterator& operator++() {
    ++it_;
    return *this;
  }

  unary_transform_iterator  operator++(int) {
    unary_transform_iterator tmp(*this);
    ++it_;
    return tmp;
  }

  unary_transform_iterator& operator--() {
    --it_;
    return *this;
  }

  unary_transform_iterator  operator--(int) {
    unary_transform_iterator tmp(*this);
    --it_;
    return tmp;
  }

  unary_transform_iterator  operator+ (difference_type n) const {
    return unary_transform_iterator(it_ + n, f_);
  }

  unary_transform_iterator& operator+=(difference_type n) {
    it_ += n;
    return *this;
  }

  unary_transform_iterator  operator- (difference_type n) const {
    return unary_transform_iterator(it_ - n, f_);
  }

  unary_transform_iterator& operator-=(difference_type n) {
    it_ -= n;
    return *this;
  }

  reference operator[](difference_type n) const {
    return static_cast<reference>(f_(it_[n]));
  }
};

template<typename Iter1, typename Iter2, typename Functor>
inline
bool
operator==(const unary_transform_iterator<Iter1, Functor>& x,
           const unary_transform_iterator<Iter2, Functor>& y) {
  return x.base() == y.base();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
bool
operator!=(const unary_transform_iterator<Iter1, Functor>& x,
           const unary_transform_iterator<Iter2, Functor>& y) {
  return x.base() != y.base();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
bool
operator<(const unary_transform_iterator<Iter1, Functor>& x,
          const unary_transform_iterator<Iter2, Functor>& y) {
  return x.base() < y.base();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
bool
operator<=(const unary_transform_iterator<Iter1, Functor>& x,
           const unary_transform_iterator<Iter2, Functor>& y) {
  return x.base() <= y.base();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
bool
operator>(const unary_transform_iterator<Iter1, Functor>& x,
          const unary_transform_iterator<Iter2, Functor>& y) {
  return x.base() > y.base();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
bool
operator>=(const unary_transform_iterator<Iter1, Functor>& x,
           const unary_transform_iterator<Iter2, Functor>& y) {
  return x.base() >= y.base();
}

template<typename Iter1, typename Iter2, typename Functor>
inline
auto
operator-(const unary_transform_iterator<Iter1, Functor>& x,
          const unary_transform_iterator<Iter2, Functor>& y)
    -> decltype(x.base() - y.base()) {
  return x.base() - y.base();
}

template<typename Iter, typename Functor>
inline
unary_transform_iterator<Iter, Functor>
operator+(typename
          unary_transform_iterator<Iter, Functor>::difference_type n,
          const unary_transform_iterator<Iter, Functor>& x) {
  return unary_transform_iterator<Iter, Functor>(x.base() + n, x.functor());
}

template<typename Iter, typename Functor>
inline
unary_transform_iterator<Iter, Functor>
make_unary_transform_iterator(const Iter& it, const Functor& f) {
  return unary_transform_iterator<Iter, Functor>(it, f);
}

}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_UNARY_TRANSFORM_ITERATOR_H_
