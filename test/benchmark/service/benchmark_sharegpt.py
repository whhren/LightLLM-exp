# Adapted from benchmarks/benchmark_serving.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Union
import os
import json
import aiohttp
import numpy as np
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm.asyncio import tqdm

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "auto",
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, *args, **kwargs)
    except TypeError as e:
        err_msg = "Failed to load the tokenizer. {e}"
        raise RuntimeError(err_msg) from e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        pass
    return tokenizer


# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    max_history_turns: int = 6,
    max_total_tokens: int = 16384,
) -> List[Tuple[List[dict], str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with at least 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= max_history_turns]

    print("read data set finish")
    dataset = dataset[: num_requests * 3]

    def to_openai_role(role_value: str) -> str:
        lower_value = role_value.lower()
        if lower_value in ["human", "user", "system"]:
            return "user" if lower_value != "system" else "system"
        return "assistant"

    # Build messages and targets
    built_examples: List[Tuple[List[dict], str]] = []
    for data in dataset:
        convs = data.get("conversations", [])
        if not convs:
            continue
        # Find the last assistant turn to be used as the completion target
        last_assistant_idx = -1
        for idx in range(len(convs) - 1, -1, -1):
            role_val = convs[idx].get("from") or convs[idx].get("role") or "assistant"
            if to_openai_role(role_val) == "assistant":
                last_assistant_idx = idx
                break
        if last_assistant_idx <= 0:
            # Need at least one prompt message before the assistant response
            continue
        # Determine how many turns of history to keep before the target assistant turn
        start_idx = max(0, last_assistant_idx - max_history_turns)
        context_convs = convs[start_idx:last_assistant_idx]
        completion_text = convs[last_assistant_idx].get("value") or convs[last_assistant_idx].get("content") or ""
        if not completion_text:
            continue
        messages: List[dict] = []
        for turn in context_convs:
            role_val = turn.get("from") or turn.get("role") or "user"
            content_val = turn.get("value") or turn.get("content") or ""
            if not content_val:
                continue
            messages.append({"role": to_openai_role(role_val), "content": content_val})
        if not messages:
            continue
        built_examples.append((messages, completion_text))

    # Render prompts using chat template when possible
    rendered_prompts: List[str] = []
    for messages, _ in built_examples:
        rendered_text = None
        try:
            # Prefer using the tokenizer's chat template
            rendered_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback rendering if chat template is unavailable
            parts = []
            for m in messages:
                parts.append(f"{m['role']}: {m['content']}")
            parts.append("assistant:")
            rendered_text = "\n".join(parts)
        rendered_prompts.append(rendered_text)

    # Tokenize the prompts and completions.
    prompt_token_ids = tokenizer(rendered_prompts).input_ids if rendered_prompts else []
    completion_texts = [completion for _, completion in built_examples]
    completion_token_ids = tokenizer(completion_texts).input_ids if completion_texts else []

    tokenized_dataset: List[Tuple[List[dict], str, int, int]] = []
    for i in range(len(built_examples)):
        messages, _ = built_examples[i]
        prompt_len = len(prompt_token_ids[i])
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((messages, rendered_prompts[i], prompt_len, output_len))

    # Filter out too long or too short sequences.
    filtered_dataset: List[Tuple[List[dict], str, int, int]] = []
    for messages, rendered_prompt, prompt_len, output_len in tokenized_dataset:
        if prompt_len < 4 or output_len < 4:
            continue
        if (prompt_len + output_len) >= max_total_tokens:
            continue
        filtered_dataset.append((messages, rendered_prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = filtered_dataset[:num_requests]
    sum_len = 0
    for _, _, prompt_len, output_len in sampled_requests:
        sum_len += prompt_len + output_len
    print("total tokens:", sum_len)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[List[dict], str, int, int]],
    request_rate: float,
    concurrency: int = None,
) -> AsyncGenerator[Tuple[List[dict], str, int, int], None]:
    input_requests = iter(input_requests)

    if concurrency is not None:
        # Concurrency-based request generation
        # This generator will be consumed by the benchmark function
        # which will manage the concurrency
        for request in input_requests:
            yield request
    else:
        # Rate-based request generation (original logic)
        for request in input_requests:
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue
            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)


async def send_request(
    messages: List[dict],
    rendered_prompt: str,
    prompt_len: int,
    output_len: int,
    use_openai_api: bool,
    port: int,
    pbar=None,
) -> None:
    if use_openai_api:
        # Use OpenAI API to send the request.
        # Use local server to send the request.
        request_start_time = time.time()
        headers = {"Content-Type": "application/json", "User-Agent": "Benchmark Client"}
        url = f"http://localhost:{port}/v1/chat/completions"

        data = {
            "model": "DeepSeek-R1",
            "messages": messages,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 0,
            "stream": True,
            "ignore_eos": False,
            "max_tokens": output_len,
        }
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        receive_n = 1

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                text = ""
                start_time = time.time()
                is_first = True
                async for chunk, _ in response.content.iter_chunks():
                    now_time = time.time()
                    delta_time = now_time - start_time
                    if is_first:
                        is_first = False
                        ttft = delta_time
                    text += json.loads(chunk.decode("utf-8")[6:])["choices"][0]["delta"].get("content", "")
                    if delta_time < 0.005:
                        receive_n += 1
                    chunks.append(delta_time)
                    start_time = now_time
        # print("messages", messages)
        # print("text", text)

    else:
        # Use local server to send the request.
        request_start_time = time.time()
        headers = {"Content-Type": "application/json", "User-Agent": "Benchmark Client"}
        url = f"http://localhost:{port}/generate_stream"

        data = {
            "inputs": rendered_prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
            },
        }

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            receive_n = 0
            text = ""
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                start_time = time.time()
                is_first = True
                async for chunk, _ in response.content.iter_chunks():
                    now_time = time.time()
                    delta_time = now_time - start_time
                    if is_first:
                        is_first = False
                        ttft = delta_time
                    if delta_time < 0.005:
                        receive_n += 1
                    chunks.append(chunk)
                    text += json.loads(chunk.decode("utf-8")[5:])["token"]["text"]
                    start_time = now_time

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency, ttft))

    # Update progress bar if provided
    if pbar:
        pbar.update(1)


async def benchmark(
    input_requests: List[Tuple[List[dict], str, int, int]],
    request_rate: float,
    use_openai_api: bool = False,
    concurrency: int = None,
    port: int = 8080,
) -> None:
    total_requests = len(input_requests)

    # Create progress bar
    pbar = tqdm(total=total_requests, desc="Processing requests", unit="req")

    if concurrency is not None:
        # Concurrency-based processing
        semaphore = asyncio.Semaphore(concurrency)
        tasks: List[asyncio.Task] = []

        async def send_with_semaphore(messages, rendered_prompt, prompt_len, output_len):
            async with semaphore:
                await send_request(messages, rendered_prompt, prompt_len, output_len, use_openai_api, port, pbar)

        async for request in get_request(input_requests, request_rate, concurrency):
            messages, rendered_prompt, prompt_len, output_len = request
            task = asyncio.create_task(send_with_semaphore(messages, rendered_prompt, prompt_len, output_len))
            tasks.append(task)

        await asyncio.gather(*tasks)
    else:
        # Rate-based processing (original logic)
        tasks: List[asyncio.Task] = []
        async for request in get_request(input_requests, request_rate, concurrency):
            messages, rendered_prompt, prompt_len, output_len = request
            task = asyncio.create_task(
                send_request(messages, rendered_prompt, prompt_len, output_len, use_openai_api, port, pbar)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer, "slow")
    input_requests = sample_requests(
        args.dataset,
        args.num_prompts,
        tokenizer,
        args.history_turns,
        args.max_total_tokens
    )
    benchmark_start_time = time.time()
    asyncio.run(
        benchmark(
            input_requests,
            args.request_rate,
            args.use_openai_api,
            args.concurrency,
            args.port
        )
    )
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    # =========================
    # 统计实验数据
    # =========================
    avg_latency = np.mean(
        [latency for _, _, latency, _ in REQUEST_LATENCY]
    )

    avg_time_to_first_token = np.mean(
        [ttft for _, _, _, ttft in REQUEST_LATENCY]
    )

    avg_per_token_latency = (
        np.mean(
            [
                latency / (prompt_len + output_len)
                for prompt_len, output_len, latency, _ in REQUEST_LATENCY
            ]
        ) * 1000
    )

    avg_inter_token_latency = (
        np.mean(
            [
                (latency - ttft) / (output_len - 1)
                for _, output_len, latency, ttft in REQUEST_LATENCY
                if output_len > 1
            ]
        )
        * 1000
    )
    # tail latency
    latencies = np.array(
        [x[2] for x in REQUEST_LATENCY]
    )
    p10_latency = np.percentile(latencies, 10)
    p20_latency = np.percentile(latencies, 20)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)

    # token长度统计
    input_lens = [
        x[0] for x in REQUEST_LATENCY
    ]
    output_lens = [
        x[1] for x in REQUEST_LATENCY
    ]
    total_lens = [
        x[0] + x[1]
        for x in REQUEST_LATENCY
    ]
    hit_max_tokens = sum(
        1 for x in total_lens
        if x >= args.max_total_tokens
    )
    # =========================
    # 保存结果
    # =========================

    os.makedirs("results", exist_ok=True)

    filename = (
        f"results/"
        f"fcfs_"
        f"rate{int(args.request_rate)}_"
        f"req{args.num_prompts}.json"
    )


    result = {

        # 实验配置
        "config": {
            "request_rate": args.request_rate,
            "num_prompts": args.num_prompts,
            "history_turns": args.history_turns,
            "max_total_tokens": args.max_total_tokens,
        },


        # 总体指标
        "summary": {

            "total_time": benchmark_time,
            "throughput": (
                len(REQUEST_LATENCY)
                / benchmark_time
            ),
            "avg_latency": float(avg_latency),
            "avg_ttft": float(
                avg_time_to_first_token
            ),
            "avg_latency_per_token_ms": float(
                avg_per_token_latency
            ),
            "avg_inter_token_latency_ms": float(
                avg_inter_token_latency
            ),
            # tail latency
            "p10_latency": float(p10_latency),
            "p20_latency": float(p20_latency),
            "p50_latency": float(p50_latency),
            "p90_latency": float(p90_latency),
            "p99_latency": float(p99_latency),

            # token情况
            "avg_input_len": float(
                np.mean(input_lens)
            ),
            "avg_output_len": float(
                np.mean(output_lens)
            ),
            "max_output_len": int(
                max(output_lens)
            ),
            # EOS/限制检测
            "hit_max_total_tokens": hit_max_tokens,

            "total_requests": len(total_lens),
        },
        # 请求级数据
        "requests": []
    }



    for i, (
        input_len,
        output_len,
        latency,
        ttft
    ) in enumerate(REQUEST_LATENCY):

        result["requests"].append(
            {
                "request_id": i,

                "input_len": input_len,

                "output_len": output_len,

                "total_len": (
                    input_len + output_len
                ),

                "latency": latency,

                "ttft": ttft,
            }
        )

    with open(filename, "w") as f:
        json.dump(
            result,
            f,
            indent=4
        )
    print(f"saved: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--use_openai_api", default=False, action="store_true", help="Use OpenAI API for requests.")
    parser.add_argument("--port", type=int, default=8080, help="Port of the API server.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Name or path of the tokenizer.")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent requests to maintain. " "Cannot be used together with --request-rate.",
    )
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
    parser.add_argument(
        "--history-turns", type=int, default=6, help="Max number of context turns before the target assistant reply."
    )
    parser.add_argument("--max-total-tokens", type=int, default=16384, help="Max total tokens (input + output).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Validate that only one of request_rate or concurrency is set
    if args.concurrency is not None and args.request_rate != float("inf"):
        raise ValueError("Cannot set both --request-rate and --concurrency. Please use only one.")

    main(args)
