/* phxrpc_ftrl_service.h

 Generated by phxrpc_pb2service from ftrl.proto

 Please DO NOT edit unless you know exactly what you are doing.

*/

#pragma once

#include "ftrl.pb.h"


class FTRLService {
  public:
    FTRLService();
    virtual ~FTRLService();

    virtual int PHXEcho(const google::protobuf::StringValue &req, google::protobuf::StringValue *resp);
    virtual int FTRL(const ftrl::FTRLRequest &req, ftrl::FTRLResult *resp);
    virtual int Feedback(const ftrl::FTRLFeedback &req, google::protobuf::Empty *resp);
};

