# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""
import copy
import time
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:

    def maybe_setup_kv_connector_with_timing(self, scheduler_output: "SchedulerOutput"):
        """
        Setup KV connector and track timing per request.
        """
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            st = time.monotonic()
            kv_connector.start_load_kv(get_forward_context())
            ed = time.monotonic()
            kv_load_time = ed - st
            logger.debug(f"KV load took {kv_load_time:.4f}s")
            
            # Track KV transfer time per request
            if hasattr(self, 'requests'):
                # Get active request IDs from scheduler output
                active_req_ids = set()
                if hasattr(scheduler_output, 'num_scheduled_tokens'):
                    active_req_ids.update(scheduler_output.num_scheduled_tokens.keys())
                
                # Distribute the KV load time among active requests
                if active_req_ids:
                    time_per_request = kv_load_time / len(active_req_ids)
                    for req_id in active_req_ids:
                        if req_id in self.requests:
                            self.requests[req_id].kv_transfer_total_time += time_per_request

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            st = time.monotonic()
            kv_connector.start_load_kv(get_forward_context())
            ed = time.monotonic()
            logger.debug(f"KV load took {ed - st:.2f}s")

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    def kv_connector_no_forward(self, scheduler_output: "SchedulerOutput",
                                vllm_config: VllmConfig) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with set_forward_context(None, vllm_config):
            self.maybe_setup_kv_connector_with_timing(scheduler_output)
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if not finished_sending and not finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        return output

    def log_kv_transfer_cumulative_time(self, req_id: str):
        """
        Log the cumulative KV transfer time for a specific request.
        This should be called when a request is finished or at decode completion.
        """
        if hasattr(self, 'requests') and req_id in self.requests:
            total_time = self.requests[req_id].kv_transfer_total_time
            if total_time > 0:
                logger.info(f"Request {req_id} total KV transfer time: {total_time:.4f}s")
