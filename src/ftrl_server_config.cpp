/* ftrl_server_config.cpp

 Generated by phxrpc_pb2server from ftrl.proto

*/

#include "ftrl_server_config.h"

#include "ftrl.pb.h"


FTRLServerConfig::FTRLServerConfig() {
}

FTRLServerConfig::~FTRLServerConfig() {
}

bool FTRLServerConfig::Read(const char *config_file) {
    bool ret{ep_server_config_.Read(config_file)};

    if (0 == strlen(ep_server_config_.GetPackageName())) {
        ep_server_config_.SetPackageName(
                ftrl::FTRLRequest::default_instance().GetDescriptor()->file()->package().c_str());
    }

    return ret;
}

phxrpc::HshaServerConfig &FTRLServerConfig::GetHshaServerConfig() {
    return ep_server_config_;
}

