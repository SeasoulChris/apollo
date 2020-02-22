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

#ifndef INCLUDE_CYBERTRON_TIME_IMPL_TIME_H_
#define INCLUDE_CYBERTRON_TIME_IMPL_TIME_H_

/*********************************************************************
** Headers
*********************************************************************/

//#include <cybertron/platform.h>
#include <iostream>
#include <cmath>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "fueling/common/record/kinglong/cybertron/time/time.h"
/*********************************************************************
** Ccybertrons Platform Headers
*********************************************************************/

#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

namespace cybertron {

template <class T, class D>
T &TimeBase<T, D>::fromNSec(uint64_t t) {
  uint64_t sec64 = 0;
  uint64_t nsec64 = t;

  normalizeSecNSec(sec64, nsec64);

  sec = (uint32_t)sec64;
  nsec = (uint32_t)nsec64;

  return *static_cast<T *>(this);
}

template <class T, class D>
D TimeBase<T, D>::operator-(const T &rhs) const {
  return D((int32_t)sec - (int32_t)rhs.sec,
           (int32_t)nsec - (int32_t)rhs.nsec);  // carry handled in ctor
}

template <class T, class D>
T TimeBase<T, D>::operator-(const D &rhs) const {
  return *static_cast<const T *>(this) + (-rhs);
}

template <class T, class D>
T TimeBase<T, D>::operator+(const D &rhs) const {
  int64_t sec_sum = (int64_t)sec + (int64_t)rhs.sec;
  int64_t nsec_sum = (int64_t)nsec + (int64_t)rhs.nsec;

  // Throws an exception if we go out of 32-bit range
  normalizeSecNSecUnsigned(sec_sum, nsec_sum);

  // now, it's safe to downcast back to uint32 bits
  return T((uint32_t)sec_sum, (uint32_t)nsec_sum);
}

template <class T, class D>
T &TimeBase<T, D>::operator+=(const D &rhs) {
  *this = *this + rhs;
  return *static_cast<T *>(this);
}

template <class T, class D>
T &TimeBase<T, D>::operator-=(const D &rhs) {
  *this += (-rhs);
  return *static_cast<T *>(this);
}

template <class T, class D>
bool TimeBase<T, D>::operator==(const T &rhs) const {
  return sec == rhs.sec && nsec == rhs.nsec;
}

template <class T, class D>
bool TimeBase<T, D>::operator<(const T &rhs) const {
  if (sec < rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec < rhs.nsec)
    return true;
  return false;
}

template <class T, class D>
bool TimeBase<T, D>::operator>(const T &rhs) const {
  if (sec > rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec > rhs.nsec)
    return true;
  return false;
}

template <class T, class D>
bool TimeBase<T, D>::operator<=(const T &rhs) const {
  if (sec < rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec <= rhs.nsec)
    return true;
  return false;
}

template <class T, class D>
bool TimeBase<T, D>::operator>=(const T &rhs) const {
  if (sec > rhs.sec)
    return true;
  else if (sec == rhs.sec && nsec >= rhs.nsec)
    return true;
  return false;
}

//  template<class T, class D>
//  boost::posix_time::ptime
//  TimeBase<T, D>::toBoost() const
//  {
//    namespace pt = boost::posix_time;
//#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
//    return pt::from_time_t(sec) + pt::nanoseconds(nsec);
//#else
//    return pt::from_time_t(sec) + pt::miccybertroneconds(nsec/1000.0);
//#endif
//  }
}

#endif  // INCLUDE_CYBERTRON_TIME_IMPL_TIME_H_
