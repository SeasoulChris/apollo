  
#include "box_iou.h"

PYBIND11_MODULE(box_iou, m) {
  m.def("rbbox_iou", &rbbox_iou<double>,
        py::return_value_policy::reference_internal, "rbbox iou",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);
  m.def("rbbox_iou", &rbbox_iou<float>,
        py::return_value_policy::reference_internal, "rbbox iou",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);
  m.def("rbbox_intersection", &rbbox_intersection<double>,
        py::return_value_policy::reference_internal, "rbbox iou",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);
  m.def("rbbox_intersection", &rbbox_intersection<float>,
        py::return_value_policy::reference_internal, "rbbox iou",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);
}