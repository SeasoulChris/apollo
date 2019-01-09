#include <gflags/gflags.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>

#include "modules/common/util/string_util.h"
#include "modules/data/fuel/proto/record_io.grpc.pb.h"

DEFINE_int32(port, 8010, "Port of the service.");

namespace apollo {
namespace data {
namespace fuel {

class RecordIOService final : public RecordIO::Service {
 public:
  grpc::Status LoadRecord(grpc::ServerContext* context, const RecordData* input,
                          RecordData* output) {
    cyber::record::RecordReader reader(input->path());

    output->set_path(input->path());
    cyber::record::RecordMessage message;
    while (reader.ReadMessage(&message)) {
    }
    return grpc::Status::OK;
  }

  grpc::Status DumpRecord(grpc::ServerContext* context, const RecordData* input,
                          RecordIOStatus* output) {
    return grpc::Status::OK;
  }
};

}  // namespace fuel
}  // namespace data
}  // namespace apollo

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  apollo::data::fuel::RecordIOService service;
  grpc::ServerBuilder builder;
  const std::string addr = apollo::common::util::StrCat("0.0.0.0:", FLAGS_port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  server->Wait();
}
