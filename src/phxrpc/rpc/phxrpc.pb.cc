// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: phxrpc/rpc/phxrpc.proto

#include "phxrpc/rpc/phxrpc.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace phxrpc {
}  // namespace phxrpc
static constexpr ::PROTOBUF_NAMESPACE_ID::Metadata* file_level_metadata_phxrpc_2frpc_2fphxrpc_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_phxrpc_2frpc_2fphxrpc_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_phxrpc_2frpc_2fphxrpc_2eproto = nullptr;
const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_phxrpc_2frpc_2fphxrpc_2eproto::offsets[1] = {};
static constexpr ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema* schemas = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::Message* const* file_default_instances = nullptr;

const char descriptor_table_protodef_phxrpc_2frpc_2fphxrpc_2eproto[] =
  "\n\027phxrpc/rpc/phxrpc.proto\022\006phxrpc\032 googl"
  "e/protobuf/descriptor.proto:/\n\005CmdID\022\036.g"
  "oogle.protobuf.MethodOptions\030\200\211z \001(\005:3\n\t"
  "OptString\022\036.google.protobuf.MethodOption"
  "s\030\201\211z \001(\t:/\n\005Usage\022\036.google.protobuf.Met"
  "hodOptions\030\202\211z \001(\tb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_deps[1] = {
  &::descriptor_table_google_2fprotobuf_2fdescriptor_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_sccs[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_once;
static bool descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto = {
  &descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_initialized, descriptor_table_protodef_phxrpc_2frpc_2fphxrpc_2eproto, "phxrpc/rpc/phxrpc.proto", 226,
  &descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_once, descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_sccs, descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto_deps, 0, 1,
  schemas, file_default_instances, TableStruct_phxrpc_2frpc_2fphxrpc_2eproto::offsets,
  file_level_metadata_phxrpc_2frpc_2fphxrpc_2eproto, 0, file_level_enum_descriptors_phxrpc_2frpc_2fphxrpc_2eproto, file_level_service_descriptors_phxrpc_2frpc_2fphxrpc_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_phxrpc_2frpc_2fphxrpc_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_phxrpc_2frpc_2fphxrpc_2eproto), true);
namespace phxrpc {
::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::google::protobuf::MethodOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::PrimitiveTypeTraits< ::PROTOBUF_NAMESPACE_ID::int32 >, 5, false >
  CmdID(kCmdIDFieldNumber, 0);
const std::string OptString_default("");
::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::google::protobuf::MethodOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::StringTypeTraits, 9, false >
  OptString(kOptStringFieldNumber, OptString_default);
const std::string Usage_default("");
::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::google::protobuf::MethodOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::StringTypeTraits, 9, false >
  Usage(kUsageFieldNumber, Usage_default);

// @@protoc_insertion_point(namespace_scope)
}  // namespace phxrpc
PROTOBUF_NAMESPACE_OPEN
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
