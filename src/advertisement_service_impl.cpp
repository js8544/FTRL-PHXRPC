/* advertisement_service_impl.cpp

 Generated by phxrpc_pb2service from ftrl.proto

*/

#include "advertisement_service_impl.h"

#include "phxrpc/file.h"

#include "advertisement_server_config.h"
#include "ftrl.pb.h"

int RequestAdId(::google::protobuf::int64 user_id){
	//request for result
	int ad_id = 0;
	return ad_id;
}

void SendFeedback(::google::protobuf::int64 ad_id){
	//send feedback for training 
}

AdvertisementServiceImpl::AdvertisementServiceImpl(ServiceArgs_t &app_args) : args_(app_args) {
}

AdvertisementServiceImpl::~AdvertisementServiceImpl() {
}

int AdvertisementServiceImpl::PHXEcho(const google::protobuf::StringValue &req, google::protobuf::StringValue *resp) {
    resp->set_value(req.value());
    return 0;
}

int AdvertisementServiceImpl::Advertisement(const ftrl::AdvertisementRequest &req, ftrl::AdvertisementResult *resp) {
    int ad_id = RequestAdId(req.user_id());//to be implemented
    resp->set_ad_id((::google::protobuf::int64) ad_id);
    return 0;
}

int AdvertisementServiceImpl::Feedback(const ftrl::AdvertisementFeedback &req, google::protobuf::Empty *resp) {
    SendFeedback(req.ad_id(),req.feedback());
    return 0;
}

