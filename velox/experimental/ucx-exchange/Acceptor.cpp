/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/ucx-exchange/Acceptor.h"

#include <cstring>

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {

/*static*/
void Acceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  if (!request || !request->isCompleted()) {
    LOG(ERROR) << "AMCallback received an invalid or incomplete request";
    return;
  }
  auto buffer =
      std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
  if (!buffer || buffer->getSize() < sizeof(HandshakeMsg)) {
    LOG(ERROR) << "AMCallback received a truncated handshake: bytes="
               << (buffer ? buffer->getSize() : 0);
    return;
  }

  HandshakeMsg handshake;
  std::memcpy(&handshake, buffer->data(), sizeof(handshake));
  if (handshake.protocolVersion != kHandshakeProtocolVersion ||
      handshake.headerSize != sizeof(HandshakeMsg) ||
      handshake.workerAddressSize == 0 ||
      handshake.workerAddressSize > kMaxWorkerAddressBytes ||
      buffer->getSize() != sizeof(HandshakeMsg) + handshake.workerAddressSize) {
    LOG(ERROR) << "AMCallback received an invalid handshake: version="
               << handshake.protocolVersion
               << " headerBytes=" << handshake.headerSize
               << " addressBytes=" << handshake.workerAddressSize
               << " totalBytes=" << buffer->getSize();
    return;
  }
  if (std::memchr(handshake.taskId, '\0', sizeof(handshake.taskId)) ==
      nullptr) {
    LOG(ERROR) << "AMCallback received a task ID without a terminator";
    return;
  }

  const auto* addressData =
      static_cast<const char*>(buffer->data()) + sizeof(HandshakeMsg);
  const std::string_view workerAddress{
      addressData, handshake.workerAddressSize};

  // Create a exchangeServer based on the information received in the initial
  // handshake.
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();

  auto bootstrapEpRef = communicator->findEndpointRefByHandle(ep);
  if (!bootstrapEpRef) {
    LOG(ERROR) << "AMCallback could not resolve the bootstrap endpoint";
    return;
  }

  const PartitionKey key = {handshake.taskId, handshake.destination};

  // Determine if this is an intra-process transfer by comparing the source's
  // workerId with our Communicator's workerId. A match means both source and
  // server are in the same Communicator singleton (same process), so
  // IntraNodeTransferRegistry (in-process std::promise/future) can be used.
  bool isIntraNodeTransfer =
      cudf_velox::CudfConfig::getInstance().intraNodeExchange &&
      (handshake.workerId == communicator->getWorkerId());

  // Disable intra-node when the task is not yet initialized (placeholder
  // queue from sinks connecting before initializeTask) or when the task
  // uses broadcast mode (all destination servers share the same
  // packed_columns — the intra-node source's destructive move would
  // corrupt it for other servers).
  if (isIntraNodeTransfer) {
    if (!UcxOutputQueueManager::getInstanceRef()->canUseIntraNode(key.taskId)) {
      VLOG(2) << "[ACCEPTOR] Disabling intra-node for task " << key.taskId
              << " (not initialized or broadcast)";
      isIntraNodeTransfer = false;
    }
  }

  std::string peerIp = bootstrapEpRef->getPeerIp();
  auto payloadEpRef = bootstrapEpRef;
  try {
    payloadEpRef = communicator->assocWorkerAddressEndpointRef(
        handshake.workerId, workerAddress, peerIp);
  } catch (const std::exception& error) {
    LOG(ERROR) << "[ACCEPTOR] Failed to create worker-address payload endpoint "
               << "for task " << key.taskId << ": " << error.what()
               << ". Falling back to bootstrap endpoint.";
  }

  // Query once so worker logs record the selected payload lanes.
  payloadEpRef->usesCudaIpc();

  auto exchangeServer = UcxExchangeServer::create(
      communicator, payloadEpRef, key, isIntraNodeTransfer);

  // Add this exchangeServer to the endpoint reference.
  payloadEpRef->addCommElem(exchangeServer);

  // Register exchangeServer with communicator.
  communicator->registerCommElement(exchangeServer);
  VLOG(2) << "[ACCEPTOR] new server: " << exchangeServer->toString()
          << " peerIp=" << peerIp << " endpoint="
          << (payloadEpRef == bootstrapEpRef ? "bootstrap" : "worker-address")
          << " isIntraNodeTransfer=" << isIntraNodeTransfer;

  // Send HandshakeResponse back to the source to inform about intra-node
  // transfer. This allows the source to bypass UCXX for all subsequent data
  // transfers.
  auto response = std::make_shared<HandshakeResponse>();
  response->isIntraNodeTransfer = exchangeServer->isIntraNodeTransfer();

  uint32_t keyHash = fnv1a_32(key.toString());
  uint64_t responseTag = getHandshakeResponseTag(keyHash);

  VLOG(3) << "Sending HandshakeResponse to " << key.toString()
          << " isIntraNodeTransfer=" << response->isIntraNodeTransfer
          << " tag=" << std::hex << responseTag;

  // Fire-and-forget: we don't need to track this request completion
  [[maybe_unused]] auto sendRequest =
      payloadEpRef->endpoint_
          ->tagSendBuilder(
              response.get(), sizeof(*response), ucxx::Tag{responseTag})
          .callbackFunction([response, keyStr = key.toString()](
                                ucs_status_t status,
                                std::shared_ptr<void> arg) {
            if (status == UCS_OK) {
              VLOG(3) << "HandshakeResponse sent successfully to " << keyStr;
            } else {
              VLOG(0) << "Failed to send HandshakeResponse to " << keyStr
                      << ": " << ucs_status_string(status);
            }
          })
          .callbackData(response)
          .build();
}

// Add endpoint reference to ucp_cp -> epRef map.
void Acceptor::registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef) {
  auto epHandle = endpointRef->endpoint_->getHandle();
  auto res = handleToEndpointRef_.insert(std::pair{epHandle, endpointRef});
  VELOX_CHECK(res.second, "Endpoint handle already exists!");
}
} // namespace facebook::velox::ucx_exchange
