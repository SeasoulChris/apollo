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

#ifndef INCLUDE_CYBERTRON_COMMON_REGISTERER_H_
#define INCLUDE_CYBERTRON_COMMON_REGISTERER_H_

#include <map>
#include <string>

#include "cybertron/common/macros.h"
#include "cybertron/common/logger.h"

namespace cybertron {

// idea from boost any but make it more simple and don't use type_info.
class Any {
 public:
  Any() : _content(nullptr) {}

  template <typename ValueType>
  Any(const ValueType &value)  // NOLINT
      : _content(new Holder<ValueType>(value)) {}

  Any(const Any &other)
      : _content(other._content ? other._content->clone() : NULL) {}

  ~Any() {
    if (_content != nullptr) {
      delete _content;
      _content = nullptr;
    }
  }

  template <typename ValueType>
  ValueType *any_cast() {
    return _content ? &static_cast<Holder<ValueType> *>(_content)->_held
                    : NULL;  // NOLINT
  }

 private:
  class PlaceHolder {
   public:
    virtual ~PlaceHolder() {}
    virtual PlaceHolder *clone() const = 0;
  };

  template <typename ValueType>
  class Holder : public PlaceHolder {
   public:
    explicit Holder(const ValueType &value) : _held(value) {}
    virtual ~Holder() {}
    virtual PlaceHolder *clone() const { return new Holder(_held); }

    ValueType _held;
  };

  PlaceHolder *_content;
};

class ObjectFactory {
 public:
  ObjectFactory() {}
  virtual ~ObjectFactory() {}
  virtual Any new_instance() { return Any(); }

 private:
  DISALLOW_COPY_AND_ASSIGN(ObjectFactory);
};

typedef std::map<std::string, ObjectFactory *> FactoryMap;
typedef std::map<std::string, FactoryMap> BaseClassMap;
BaseClassMap &global_factory_map();

bool get_registered_classes(
    const std::string &base_class_name,
    std::vector<std::string> *registered_derived_classes_names);

}  // namespace cybertron

#define CYBER_REGISTER_REGISTERER(base_class)                                  \
  class base_class##Registerer {                                         \
    typedef ::cybertron::Any Any;                                        \
    typedef ::cybertron::FactoryMap FactoryMap;                          \
                                                                         \
   public:                                                               \
    static base_class *get_instance_by_name(const ::std::string &name) { \
      FactoryMap &map = ::cybertron::global_factory_map()[#base_class];  \
      FactoryMap::iterator iter = map.find(name);                        \
      if (iter == map.end()) {                                           \
        for (auto c : map) {                                             \
          LOG_INFO << "Instance:" << c.first;                           \
        }                                                                \
        LOG_ERROR << CYBERTRON_ERROR << GET_CLASS_INSTANCE_ERROR << "Get instance " << name << " failed.";              \
        return NULL;                                                     \
      }                                                                  \
      Any object = iter->second->new_instance();                         \
      return *(object.any_cast<base_class *>());                         \
    }                                                                    \
    static std::vector<base_class *> get_all_instances() {               \
      std::vector<base_class *> instances;                               \
      FactoryMap &map = ::cybertron::global_factory_map()[#base_class];  \
      instances.reserve(map.size());                                     \
      for (auto item : map) {                                            \
        Any object = item.second->new_instance();                        \
        instances.push_back(*(object.any_cast<base_class *>()));         \
      }                                                                  \
      return instances;                                                  \
    }                                                                    \
    static const ::std::string get_uniq_instance_name() {                \
      FactoryMap &map = ::cybertron::global_factory_map()[#base_class];  \
      CHECK_EQ(map.size(), 1) << map.size();                             \
      return map.begin()->first;                                         \
    }                                                                    \
    static base_class *get_uniq_instance() {                             \
      FactoryMap &map = ::cybertron::global_factory_map()[#base_class];  \
      CHECK_EQ(map.size(), 1) << map.size();                             \
      Any object = map.begin()->second->new_instance();                  \
      return *(object.any_cast<base_class *>());                         \
    }                                                                    \
    static bool is_valid(const ::std::string &name) {                    \
      FactoryMap &map = ::cybertron::global_factory_map()[#base_class];  \
      return map.find(name) != map.end();                                \
    }                                                                    \
  };

#define REGISTER_CLASS(clazz, name)                                           \
  namespace {                                                                 \
  class ObjectFactory##name : public cybertron::ObjectFactory {               \
   public:                                                                    \
    virtual ~ObjectFactory##name() {}                                         \
    virtual ::cybertron::Any new_instance() {                                 \
      return ::cybertron::Any(new name());                                    \
    }                                                                         \
  };                                                                          \
  __attribute__((constructor)) void register_factory_##name() {               \
    ::cybertron::FactoryMap &map = ::cybertron::global_factory_map()[#clazz]; \
    if (map.find(#name) == map.end()) map[#name] = new ObjectFactory##name(); \
  }                                                                           \
  }

#endif  // INCLUDE_CYBERTRON_COMMON_REGISTERER_H_
