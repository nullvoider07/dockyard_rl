"""vLLM async-engine generation worker for Project Dockyard.

The async engine is used when expose_http_server=True in VllmConfig
(OpenAI-compatible endpoint for coding agents) or when vllm_cfg.async_engine
is explicitly set to True.
"""

import asyncio
import gc
import logging
import os
import threading
import uuid
from typing import Any, AsyncGenerator, Optional, cast

import torch

logger = logging.getLogger(__name__)

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from dockyard_rl.models.generation.vllm.config import VllmConfig
from .utils import (
    format_prompt_for_vllm_generation,
)
from dockyard_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    wrap_with_nvtx_name,
)

# Async-engine worker implementation
class VllmAsyncGenerationWorkerImpl(BaseVllmGenerationWorker):
    """vLLM async-engine generation worker.

    Uses vllm.AsyncLLMEngine instead of vllm.LLM.  The async engine is
    mandatory when expose_http_server=True because vLLM's OpenAI-compatible
    HTTP server requires an AsyncLLMEngine under the hood.

    The event loop runs in a dedicated daemon thread so Ray actor method calls
    (which are synchronous from the driver's perspective) can await coroutines
    without blocking the Ray event loop.
    """

    llm: Any  # AsyncLLMEngine at runtime; set by _create_engine

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        """Construct an AsyncLLMEngine.

        AsyncLLMEngine does not accept all the same kwargs as vllm.LLM;
        unsupported keys are stripped before forwarding.
        """
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        # Keys accepted by LLM but not AsyncEngineArgs.
        _async_unsupported = {
            "disable_log_stats",
            "logprobs_mode",
            "served_model_name",
        }
        async_kwargs = {
            k: v for k, v in llm_kwargs.items()
            if k not in _async_unsupported
        }

        engine_args = AsyncEngineArgs(**async_kwargs)
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        # Dedicated event loop in a daemon thread.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="vllm-async-loop",
        )
        self._loop_thread.start()

    def _run_async(self, coro):
        """Submit a coroutine to the worker event loop and block until done."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def post_init(self) -> None:
        self.vllm_device_ids = self.report_device_id_async()
        if self.cfg["vllm_cfg"].get("expose_http_server", False):
            self._start_http_server()

    # HTTP server (optional)
    def _start_http_server(self) -> None:
        """Launch vLLM's OpenAI-compatible HTTP server in a background thread.

        Port is read from DOCKYARD_VLLM_HTTP_PORT (default 8000).
        The server is started in a daemon thread so it doesn't block the
        Ray actor lifecycle.
        """
        from vllm.entrypoints.openai.api_server import (
            build_app,
            init_app_state,
        )
        import uvicorn

        port = int(os.environ.get("DOCKYARD_VLLM_HTTP_PORT", "8000"))
        serving_kwargs = self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", {}
        )

        async def _serve():
            app = build_app(serving_kwargs)  # type: ignore[arg-type]
            await init_app_state(
                engine=self.llm,  # type: ignore[call-arg]
                engine_config=self.llm.engine_config,  # type: ignore[call-arg]
                state=app.state,
                args=type("_Args", (), {
                    "model":               self.model_name,
                    "served_model_name":  [self.model_name],
                    **serving_kwargs,
                })(),
            )
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            await server.serve()

        def _run():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_serve())

        # Register custom parser plugin files before the server starts, so a
        # parser selected by name resolves. Then serialise engine input-socket
        # sends before the HTTP server thread starts: the server and the
        # async-GRPO weight-update path both send on the engine's zmq input
        # socket concurrently.
        self._register_parser_plugins()
        self._install_engine_input_socket_lock()

        t = threading.Thread(target=_run, daemon=True, name="vllm-http-server")
        t.start()

    def _register_parser_plugins(self) -> None:
        """Register custom tool / reasoning parser plugin files with vLLM.

        A parser is selected by name at serving time (tool_call_parser via
        http_server_serving_chat_kwargs; the reasoning parser via the engine
        config). That named parser only exists once its plugin file has been
        imported — which runs the file's @register_module decorator. Passing
        the plugin path as a serving arg does not trigger this (init_app_state
        reads tool_call_parser/enable_auto_tool_choice, not the plugin path),
        so register the plugin file explicitly.

        Guarded against vLLM module-path drift: if the parser manager cannot be
        imported the plugin is skipped with a warning. A bad plugin path is
        left to raise — that is a real misconfiguration, not version drift.
        """
        tool_parser_plugin = self.cfg["vllm_cfg"].get("tool_parser_plugin")
        if tool_parser_plugin:
            try:
                from vllm.tool_parsers.abstract_tool_parser import (
                    ToolParserManager,
                )
            except ImportError as exc:
                logger.warning(
                    "vLLM ToolParserManager unavailable; cannot register "
                    "tool_parser_plugin %r: %s", tool_parser_plugin, exc,
                )
            else:
                ToolParserManager.import_tool_parser(tool_parser_plugin)

        reasoning_parser_plugin = self.cfg["vllm_cfg"].get(
            "reasoning_parser_plugin"
        )
        if reasoning_parser_plugin:
            try:
                from vllm.reasoning.abs_reasoning_parsers import (
                    ReasoningParserManager,
                )
            except ImportError as exc:
                logger.warning(
                    "vLLM ReasoningParserManager unavailable; cannot register "
                    "reasoning_parser_plugin %r: %s",
                    reasoning_parser_plugin, exc,
                )
            else:
                ReasoningParserManager.import_reasoning_parser(
                    reasoning_parser_plugin
                )

    def _install_engine_input_socket_lock(self) -> None:
        """Serialise sends on the engine's AsyncMPClient input socket across
        OS threads, preventing a race that can block the vLLM engine during
        in-flight weight updates in async GRPO.

        The socket is a private vLLM internal; if the attribute path is not
        present in the running vLLM build the lock is skipped with a warning
        rather than crashing engine startup, since this is a best-effort
        concurrency guard.
        """
        try:
            shadow_sock = self.llm.engine_core.input_socket._shadow_sock
        except AttributeError:
            logger.warning(
                "Could not locate the engine input socket "
                "(llm.engine_core.input_socket._shadow_sock); skipping the "
                "send_multipart serialisation lock. Concurrent HTTP-server and "
                "weight-update sends are not serialised on this vLLM build."
            )
            return

        lock = threading.Lock()
        original_send_multipart = shadow_sock.send_multipart

        def locked_send_multipart(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return original_send_multipart(*args, **kwargs)

        # Replace the bound method on this socket instance only; other zmq
        # sockets in the process are unaffected.
        shadow_sock.send_multipart = locked_send_multipart  # type: ignore[assignment]

    # Liveness check (for Ray)
    def is_alive(self) -> bool:
        return True

    # Collective initialisation and generation
    def init_collective(
        self,
        rank_prefix:      int,
        ip:               str,
        port:             int,
        world_size:       int,
        train_world_size: int,
    ) -> None:
        async def _inner():
            await self.llm.collective_rpc(
                "init_collective",
                args=(rank_prefix, ip, port, world_size, train_world_size),
            )
        self._run_async(_inner())

    # Generation
    @wrap_with_nvtx_name("vllm_async_generation_worker/generate")
    def generate(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Synchronous-wrapper around async generation."""
        return self._run_async(self._generate_async(data, greedy))

    async def _generate_async(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if len(data["input_ids"]) == 0:
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids":                 torch.zeros((0, 0), dtype=torch.long),
                    "logprobs":                   torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths":         torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths":  torch.zeros(0, dtype=torch.long),
                    "truncated":                  torch.zeros(0, dtype=torch.bool),
                }
            )

        input_ids     = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop    = data.get("stop_strings", [])
        stop_strings  = self._merge_stop_strings(batch_stop)
        sampling_params = self._build_sampling_params(
            greedy=greedy, stop_strings=stop_strings
        )

        verify_right_padding(data, pad_value=self.cfg.get("_pad_token_id", 0))
        padded_input_length = input_ids.size(1)
        prompts = format_prompt_for_vllm_generation(
            data,
            allow_multimodal_inputs=self.cfg.get("allow_multimodal_inputs", False),
            max_image_pixels=self.cfg.get("max_image_pixels"),
            max_images_per_sample=self.cfg.get("max_images_per_sample"),
        )

        # Fire all async generation requests and collect results in order.
        request_ids = [f"req-{i}" for i in range(len(prompts))]
        gens = [
            self.llm.generate(
                prompt,  # type: ignore[arg-type]
                sampling_params, request_id=rid
            )
            for prompt, rid in zip(prompts, request_ids)
        ]

        outputs = []
        for gen in gens:
            final = None
            async for o in gen:
                final = o
            outputs.append(final)

        output_ids_list:           list[torch.Tensor] = []
        logprobs_list:             list[torch.Tensor] = []
        generation_lengths:        list[int]          = []
        unpadded_sequence_lengths: list[int]          = []
        truncated_list:            list[bool]         = []

        max_gen = max(len(o.outputs[0].token_ids) for o in outputs)

        for i, output in enumerate(outputs):
            seq_len = int(input_lengths[i])
            gen     = output.outputs[0]
            gen_ids = list(gen.token_ids)
            n_gen   = len(gen_ids)

            total_len = padded_input_length + max_gen
            full_output = torch.full(
                (total_len,), self.cfg.get("_pad_token_id", 0),
                dtype=input_ids.dtype,
            )
            full_output[:seq_len] = input_ids[i, :seq_len]
            full_output[seq_len: seq_len + n_gen] = torch.tensor(gen_ids)
            output_ids_list.append(full_output)

            full_logprobs = torch.zeros(total_len, dtype=torch.float32)
            if hasattr(gen, "logprobs") and gen.logprobs:
                try:
                    for idx, lp_dict in enumerate(gen.logprobs):
                        if lp_dict:
                            position = seq_len + idx
                            full_logprobs[position] = next(
                                iter(lp_dict.items())
                            )[1].logprob
                except Exception:
                    import traceback
                    traceback.print_exc()
            logprobs_list.append(full_logprobs)

            generation_lengths.append(n_gen)
            unpadded_sequence_lengths.append(seq_len + n_gen)
            truncated_list.append(gen.finish_reason == "length")

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids":  torch.stack(output_ids_list),
                "logprobs":    torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(
                    generation_lengths, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths, dtype=torch.long
                ),
                "truncated": torch.tensor(truncated_list, dtype=torch.bool),
            }
        )

    async def generate_async(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Stream generation, yielding each sample as soon as it completes.

        Restricted to single-sample inputs: the driver shards batches into
        per-sample calls and round-robins them across DP leaders, so the
        streaming/concurrency lives at the rollout layer. Yields
        (sample_idx, BatchedDataDict) per sample.
        """
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled "
                "in the vLLM config."
            )

        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg.get("_pad_token_id", 0))

        input_ids_batch     = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size          = input_ids_batch.shape[0]

        assert batch_size == 1, (
            "generate_async is restricted to single samples, but received "
            f"batch_size={batch_size}. Batch outside this method."
        )

        batch_stop_strings = data.get("stop_strings", [[] for _ in range(batch_size)])
        # Per-sample constrained-decoding specs (structured tool-use, fork 2);
        # absent on the free-generation path → byte-identical sampling params.
        batch_structured_specs = data.get("structured_spec")
        pad_token_id       = self.cfg.get("_pad_token_id", 0)

        async def process_single_sample(sample_idx: int):
            current_input_actual_length = int(input_lengths_batch[sample_idx].item())
            prompt = format_prompt_for_vllm_generation(
                data,
                sample_idx,
                allow_multimodal_inputs=self.cfg.get("allow_multimodal_inputs", False),
                max_image_pixels=self.cfg.get("max_image_pixels"),
                max_images_per_sample=self.cfg.get("max_images_per_sample"),
            )

            per_sample_stop = None
            if batch_stop_strings and sample_idx < len(batch_stop_strings):
                per_sample_stop = batch_stop_strings[sample_idx]
            stop_strings = self._merge_stop_strings(
                [per_sample_stop] if per_sample_stop else None
            )

            remaining_ctx = (
                self.cfg["vllm_cfg"]["max_model_len"] - current_input_actual_length
            )
            allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

            original_input_ids_row = input_ids_batch[sample_idx]

            if allowed_new_tokens == 0:
                output_ids = original_input_ids_row[
                    :current_input_actual_length
                ].unsqueeze(0)
                return sample_idx, BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids,
                        "logprobs": torch.zeros(
                            (1, current_input_actual_length),
                            dtype=torch.float32,
                            device=original_input_ids_row.device,
                        ),
                        "generation_lengths": torch.tensor(
                            [0], dtype=torch.long, device=original_input_ids_row.device
                        ),
                        "unpadded_sequence_lengths": torch.tensor(
                            [current_input_actual_length],
                            dtype=torch.long,
                            device=original_input_ids_row.device,
                        ),
                        "truncated": torch.tensor(
                            [False], dtype=torch.bool, device=original_input_ids_row.device
                        ),
                    }
                )

            structured_spec = None
            if (
                batch_structured_specs is not None
                and sample_idx < len(batch_structured_specs)
            ):
                structured_spec = batch_structured_specs[sample_idx]
            sampling_params = self._build_sampling_params(
                greedy=greedy,
                stop_strings=stop_strings,
                max_new_tokens=allowed_new_tokens,
                structured_spec=structured_spec,
            )
            request_id = str(uuid.uuid4())

            final_request_output = None
            async for req_output in self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            gen          = final_request_output.outputs[0]
            gen_ids      = list(gen.token_ids)
            n_gen        = len(gen_ids)
            total_len    = current_input_actual_length + n_gen

            output_ids = torch.full(
                (total_len,), pad_token_id,
                dtype=original_input_ids_row.dtype,
                device=original_input_ids_row.device,
            )
            output_ids[:current_input_actual_length] = original_input_ids_row[
                :current_input_actual_length
            ]
            output_ids[current_input_actual_length:total_len] = torch.tensor(
                gen_ids,
                dtype=original_input_ids_row.dtype,
                device=original_input_ids_row.device,
            )

            logprobs = torch.zeros(
                (1, total_len), dtype=torch.float32, device=original_input_ids_row.device
            )
            if getattr(gen, "logprobs", None):
                for idx, lp_dict in enumerate(gen.logprobs):
                    if lp_dict and idx < n_gen:
                        token_id = gen_ids[idx]
                        if token_id in lp_dict:
                            pos = current_input_actual_length + idx
                            if pos < total_len:
                                logprobs[0, pos] = lp_dict[token_id].logprob

            return sample_idx, BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids.unsqueeze(0),
                    "logprobs": logprobs,
                    "generation_lengths": torch.tensor(
                        [n_gen], dtype=torch.long, device=original_input_ids_row.device
                    ),
                    "unpadded_sequence_lengths": torch.tensor(
                        [total_len], dtype=torch.long, device=original_input_ids_row.device
                    ),
                    "truncated": torch.tensor(
                        [gen.finish_reason == "length"],
                        dtype=torch.bool,
                        device=original_input_ids_row.device,
                    ),
                }
            )

        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]
        for completed in asyncio.as_completed(sample_tasks):
            try:
                yield await completed
            except Exception:
                for task in sample_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sample_tasks, return_exceptions=True)
                raise

    # Weight management
    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
    ) -> None:
        async def _inner():
            await self.llm.collective_rpc(
                "prepare_refit_info", args=(state_dict_info,)
            )
        self._run_async(_inner())

    @wrap_with_nvtx_name("vllm_async_worker/update_weights_from_collective")
    def update_weights_from_collective(self) -> bool:
        async def _inner():
            result = await self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
            )
            return result[0]

        try:
            ok = self._run_async(_inner())
            if not ok:
                print(f"Error: async worker failed to update weights: {ok}")
            return bool(ok)
        except Exception as e:
            print(f"Exception during async weight update: {e}")
            import traceback; traceback.print_exc()
            return False

    def report_device_id_async(self) -> list[str]:
        async def _inner():
            return await self.llm.collective_rpc(
                "report_device_id", args=tuple()
            )
        return cast(list[str], self._run_async(_inner()))

    # Lifecycle management
    def reset_prefix_cache_async(self) -> None:
        async def _inner():
            await self.llm.reset_prefix_cache()
        self._run_async(_inner())
        gc.collect()
        torch.cuda.empty_cache()

    def sleep_async(self) -> None:
        async def _inner():
            await self.llm.reset_prefix_cache()
            await self.llm.sleep(level=1)
        self._run_async(_inner())
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up_async(self, **kwargs) -> None:
        tags = kwargs.get("tags")
        async def _inner():
            await self.llm.wake_up(**({"tags": tags} if tags else {}))
        self._run_async(_inner())

    def shutdown(self) -> bool:
        try:
            async def _inner():
                await self.llm.collective_rpc("cleanup", args=tuple())
            self._run_async(_inner())
            del self.llm
            self.llm = None
            if hasattr(self, "_loop") and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"Error during async vLLM shutdown: {e}")
            return False

try:
    import ray
    from dockyard_rl.distributed.worker_group_utils import (  # type: ignore[import]
        get_nsight_config_if_pattern_matches,
    )

    @ray.remote(
        runtime_env={
            **get_nsight_config_if_pattern_matches(
                "vllm_async_generation_worker"
            )
        }
    )  # pragma: no cover
    class VllmAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
        pass

except ImportError:
    # Allows importing this module in environments where Ray is not installed
    # (e.g. local type-checking, test fixtures).
    VllmAsyncGenerationWorker = VllmAsyncGenerationWorkerImpl  # type: ignore[misc]s