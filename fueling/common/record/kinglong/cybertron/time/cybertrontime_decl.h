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

#ifndef INCLUDE_CYBERTRON_TIME_CYBERTRONTIME_DECL_H_
#define INCLUDE_CYBERTRON_TIME_CYBERTRONTIME_DECL_H_

#include "fueling/common/record/kinglong/cybertron/common/macros.h"

#ifdef ROS_BUILD_SHARED_LIBS  // cybertron is being built around shared
                              // libraries
#ifdef cybertrontime_EXPORTS  // we are building a shared lib/dll
#define INCLUDE_CYBERTRON_TIME_CYBERTRONTIME_DECL_H_
#else  // we are using shared lib/dll
#define INCLUDE_CYBERTRON_TIME_CYBERTRONTIME_DECL_H_
#endif
#else  // cybertron is being built around static libraries
#define CYBERTIME_DECL
#endif

#endif  // INCLUDE_CYBERTRON_TIME_CYBERTRONTIME_DECL_H_
