/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#ifndef INCLUDE_CYBERTRON_TIME_IMPL_DURATION_H_
#define INCLUDE_CYBERTRON_TIME_IMPL_DURATION_H_

#include <boost/date_time/posix_time/posix_time_types.hpp>

#include "fueling/common/record/kinglong/cybertron/time/rate.h"
namespace cybertron {
//
// DurationBase template member function implementation
//
template <class T>
DurationBase<T>::DurationBase(int32_t _sec, int32_t _nsec)
    : sec(_sec), nsec(_nsec) {
  normalizeSecNSecSigned(sec, nsec);
}

template <class T>
T &DurationBase<T>::fromSec(double d) {
  sec = (int32_t)floor(d);
  nsec = (int32_t)((d - (double)sec) * 1000000000);
  return *static_cast<T *>(this);
}

template <class T>
T &DurationBase<T>::fromNSec(int64_t t) {
  sec = (int32_t)(t / 1000000000);
  nsec = (int32_t)(t % 1000000000);

  normalizeSecNSecSigned(sec, nsec);

  return *static_cast<T *>(this);
}

template <class T>
T DurationBase<T>::operator+(const T &rhs) const {
  return T(sec + rhs.sec, nsec + rhs.nsec);
}

template <class T>
T DurationBase<T>::operator*(double scale) const {
  return T(toSec() * scale);
}

template <class T>
T DurationBase<T>::operator-(const T &rhs) const {
  return T(sec - rhs.sec, nsec - rhs.nsec);
}

template <class T>
T DurationBase<T>::operator-() const {
  return T(-sec, -nsec);
}

template <class T>
T &DurationBase<T>::operator+=(const T &rhs) {
  *this = *this + rhs;
  return *static_cast<T *>(this);
}

template <class T>
T &DurationBase<T>::operator-=(const T &rhs) {
  *this += (-rhs);
  return *static_cast<T *>(this);
}

template <class T>
T &DurationBase<T>::operator*=(double scale) {
  fromSec(toSec() * scale);
  return *static_cast<T *>(this);
}

template <class T>
bool DurationBase<T>::operator<(const T &rhs) const {
  if (sec < rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec < rhs.nsec)
    return true;
  return false;
}

template <class T>
bool DurationBase<T>::operator>(const T &rhs) const {
  if (sec > rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec > rhs.nsec)
    return true;
  return false;
}

template <class T>
bool DurationBase<T>::operator<=(const T &rhs) const {
  if (sec < rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec <= rhs.nsec)
    return true;
  return false;
}

template <class T>
bool DurationBase<T>::operator>=(const T &rhs) const {
  if (sec > rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec >= rhs.nsec)
    return true;
  return false;
}

template <class T>
bool DurationBase<T>::operator==(const T &rhs) const {
  return sec == rhs.sec && nsec == rhs.nsec;
}

template <class T>
bool DurationBase<T>::isZero() const {
  return sec == 0 && nsec == 0;
}

//  template <class T>
//  boost::posix_time::time_duration
//  DurationBase<T>::toBoost() const
//  {
//    namespace bt = boost::posix_time;
//#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
//    return bt::seconds(sec) + bt::nanoseconds(nsec);
//#else
//    return bt::seconds(sec) + bt::miccybertroneconds(nsec/1000.0);
//#endif
//  }
}
#endif
