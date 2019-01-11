#include <gflags/gflags.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>

#include "cyber/common/log.h"
#include "cyber/record/file/record_file_reader.h"
#include "cyber/record/file/record_file_writer.h"
#include "modules/common/util/string_util.h"
#include "modules/data/fuel/proto/record_io.grpc.pb.h"

DEFINE_int32(port, 8010, "Port of the service.");

namespace apollo {
namespace data {
namespace fuel {

using apollo::cyber::proto::ChunkBody;
using apollo::cyber::proto::SectionType;

class RecordIOService final : public RecordIO::Service {
 public:
  grpc::Status LoadRecord(grpc::ServerContext* context, const RecordPath* input,
                          grpc::ServerWriter<ChunkBody>* output) override {
    apollo::cyber::record::RecordFileReader reader;
    CHECK(reader.Open(input->path())) << "Cannot open " << input->path();

    apollo::cyber::record::Section section;
    ChunkBody chunk;
    // Read until the last section, which is SECTION_INDEX.
    while (reader.ReadSection(&section) &&
           section.type != SectionType::SECTION_INDEX) {
      if (section.type == SectionType::SECTION_CHUNK_BODY) {
        if (reader.ReadSection<ChunkBody>(section.size, &chunk)) {
          CHECK(output->Write(chunk));
        }
      } else {
        reader.SkipSection(section.size);
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status DumpRecord(grpc::ServerContext* context,
                          grpc::ServerReader<RecordData>* input,
                          None* output) override {
    apollo::cyber::record::RecordFileWriter writer;
    bool inited = false;

    RecordData data;
    while (input->Read(&data)) {
      if (!inited) {
        CHECK(writer.Open(data.path())) << "Cannot open " << data.path();
        inited = true;
      }
      for (const auto& message : data.messages()) {
        CHECK(writer.WriteMessage(message));
      }
    }

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
  return 0;
}
