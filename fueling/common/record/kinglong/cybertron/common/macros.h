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

#ifndef INCLUDE_CYBERTRON_COMMON_MACROS_H_
#define INCLUDE_CYBERTRON_COMMON_MACROS_H_

#include <memory>
#include <utility>

#define UNUSED(param) (void) param

#define SMART_PTR_DEFINITIONS(...)    \
  SHARED_PTR_DEFINITIONS(__VA_ARGS__) \
  WEAK_PTR_DEFINITIONS(__VA_ARGS__)   \
  UNIQUE_PTR_DEFINITIONS(__VA_ARGS__)

#define SMART_PTR_DEFINITIONS_NOT_COPYABLE(...) \
  SHARED_PTR_DEFINITIONS(__VA_ARGS__)           \
  WEAK_PTR_DEFINITIONS(__VA_ARGS__)             \
  __UNIQUE_PTR_ALIAS(__VA_ARGS__)

#define SMART_PTR_ALIASES_ONLY(...) \
  __SHARED_PTR_ALIAS(__VA_ARGS__)   \
  __WEAK_PTR_ALIAS(__VA_ARGS__)     \
  __MAKE_SHARED_DEFINITION(__VA_ARGS__)

#define __SHARED_PTR_ALIAS(...)                   \
  using SharedPtr = std::shared_ptr<__VA_ARGS__>; \
  using ConstSharedPtr = std::shared_ptr<const __VA_ARGS__>;

#define __MAKE_SHARED_DEFINITION(...)                                  \
  template <typename... Args>                                          \
  static std::shared_ptr<__VA_ARGS__> make_shared(Args &&... args) {   \
    return std::make_shared<__VA_ARGS__>(std::forward<Args>(args)...); \
  }

/// Defines aliases and static functions for using the Class with shared_ptrs.
#define SHARED_PTR_DEFINITIONS(...) \
  __SHARED_PTR_ALIAS(__VA_ARGS__)   \
  __MAKE_SHARED_DEFINITION(__VA_ARGS__)

#define __WEAK_PTR_ALIAS(...)                 \
  using WeakPtr = std::weak_ptr<__VA_ARGS__>; \
  using ConstWeakPtr = std::weak_ptr<const __VA_ARGS__>;

/// Defines aliases and static functions for using the Class with weak_ptrs.
#define WEAK_PTR_DEFINITIONS(...) __WEAK_PTR_ALIAS(__VA_ARGS__)

#define __UNIQUE_PTR_ALIAS(...) using UniquePtr = std::unique_ptr<__VA_ARGS__>;

#define __MAKE_UNIQUE_DEFINITION(...)                                \
  template <typename... Args>                                        \
  static std::unique_ptr<__VA_ARGS__> make_unique(Args &&... args) { \
    return std::unique_ptr<__VA_ARGS__>(                             \
        new __VA_ARGS__(std::forward<Args>(args)...));               \
  }

/// Defines aliases and static functions for using the Class with unique_ptrs.
#define UNIQUE_PTR_DEFINITIONS(...) \
  __UNIQUE_PTR_ALIAS(__VA_ARGS__)   \
  __MAKE_UNIQUE_DEFINITION(__VA_ARGS__)

#define STRING_JOIN(arg1, arg2) DO_STRING_JOIN(arg1, arg2)
#define DO_STRING_JOIN(arg1, arg2) arg1##arg2

// There must be many copy-paste versions of these macros which are same
// things, undefine them to avoid conflict.
#undef DISALLOW_COPY
#undef DISALLOW_ASSIGN
#undef DISALLOW_COPY_AND_ASSIGN
#undef DISALLOW_IMPLICIT_CONSTRUCTORS

// Put this in the private: declarations for a class to be uncopyable.
#define DISALLOW_COPY(...) __VA_ARGS__(const __VA_ARGS__ &) = delete;

// Put this in the private: declarations for a class to be unassignable.
#define DISALLOW_ASSIGN(...) \
  __VA_ARGS__ &operator=(const __VA_ARGS__ &) = delete;

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(classname) \
  classname(const classname &) = delete;    \
  classname &operator=(const classname &) = delete;

// A macro to disallow all the implicit constructors, namely the
// default constructor, copy constructor and operator= functions.
//
// This should be used in the private: declarations for a class
// that wants to prevent anyone from instantiating it. This is
// especially useful for classes containing only static methods.
#define DISALLOW_IMPLICIT_CONSTRUCTORS(classname) \
  classname();                                    \
  DISALLOW_COPY_AND_ASSIGN(classname)

#define DECLARE_SINGLETON(classname)                                    \
 public:                                                                \
  static const std::shared_ptr<classname> &Instance() {                 \
    static auto instance = std::shared_ptr<classname>(new classname()); \
    return instance;                                                    \
  }                                                                     \
                                                                        \
 private:                                                               \
  classname();                                                          \
  DISALLOW_COPY_AND_ASSIGN(classname)

#define DECLARE_SINGLETON_WITH_INIT(classname)          \
 public:                                                \
  static const std::shared_ptr<classname> &Instance() { \
    static std::shared_ptr<classname> instance;         \
    static std::mutex _single_mutex;                    \
    if (!instance) {                                    \
      std::lock_guard<std::mutex> lock(_single_mutex);  \
      if (!instance) {                                  \
        instance.reset(new (std::nothrow) classname);   \
        if (instance->Init() != cybertron::SUCC) {                 \
          instance.reset();                             \
          instance = nullptr;                           \
        }                                               \
      }                                                 \
    }                                                   \
    return instance;                                    \
  }                                                     \
                                                        \
 private:                                               \
  classname();                                          \
  int Init();                                           \
  DISALLOW_COPY_AND_ASSIGN(classname)

#define DECLARE_SINGLETON_MULTI_PROCESS_WITH_INIT(classname)\
 public:                                                \
  static const std::shared_ptr<classname> &Instance() { \
    static std::map<int, std::shared_ptr<classname>> instance_map;\
    static std::mutex _single_mutex;                    \
    int pid = (int)getpid();                            \
    if (instance_map.count(pid) == 0) {                 \
      std::lock_guard<std::mutex> lock(_single_mutex);  \
      std::shared_ptr<classname> instance = nullptr;    \
      instance.reset(new (std::nothrow) classname);     \
      if (instance->Init() != cybertron::SUCC) {                   \
        instance.reset();                               \
        instance = nullptr;                             \
        return nullptr;                                 \
      } else {                                          \
        instance_map[pid] = instance;                   \
      }                                                 \
    }                                                   \
    return instance_map[pid];                           \
  }                                                     \
                                                        \
 private:                                               \
  classname();                                          \
  int Init();                                           \
  DISALLOW_COPY_AND_ASSIGN(classname)

#endif  // INCLUDE_CYBERTRON_COMMON_MACROS_H_
