import asyncio
import json
from typing import Any, Dict, List

from google.genai import types
from google.genai.types import GenerateContentConfig

from app.config import ENV_PROJECT
from app.insurance_policy_processor.policy_document_handler.policy_file_manager import (
    delete_policy_file_from_gemini_future,
    upload_policy_file,
)
from app.insurance_policy_processor.policy_orchestrator.modules.gemini import (
    gemini_client,
)
from app.modules.async_redis import redis_client
from app.utils.core import pdf_to_bytes
from app.logging_config import get_logger

logger = get_logger(__name__)

MODEL_LIMITS = {
    "models/gemini-2.5-flash": {
        "tpm": ENV_PROJECT.FLASH_TPM_LIMIT,
        "rpm": ENV_PROJECT.FLASH_RPM_LIMIT,
        "window_seconds": 60,
    },
    "models/gemini-2.5-flash-lite": {
        "tpm": ENV_PROJECT.FLASH_LIGHT_TPM_LIMIT,
        "rpm": ENV_PROJECT.FLASH_LIGHT_RPM_LIMIT,
        "window_seconds": 60,
    },
}

KEY_WORK_QUEUE = "gemini:work_queue"
KEY_PRIORITY_QUEUE = "gemini:priority_queue"  # For retries and urgent tasks
KEY_RESULTS_HASH = "gemini:results"
KEY_TPM_PREFIX = "gemini:tpm_available"
KEY_RPM_PREFIX = "gemini:rpm_available"


async def submit_gemini_task(
    task_id: str,
    model: str,
    contents: Any,
    config: Dict,
    estimated_tokens: int,
    file_path: str = None,
    file_processing: str = None,
    mime_type: str = None,
    text_contents: List[str] = None,
    priority: bool = False,
) -> None:
    # serializes a task and adds it to the Redis work queue
    logger.debug(
        f"Submitting Gemini task {task_id} with model {model}, estimated tokens: {estimated_tokens}, priority: {priority}"
    )

    payload = {
        "model": model,
        "contents": contents,
        "config": config,
        "estimated_tokens": estimated_tokens,
    }

    # add file processing parameters if provided
    if file_path and file_processing:
        payload.update(
            {
                "file_processing": file_processing,
                "file_path": file_path,
                "mime_type": mime_type,
                "text_contents": text_contents,
            }
        )

    task_data = {
        "task_id": task_id,
        "payload": payload,
        "retry_count": 0,
        "submitted_at": int((await redis_client.client.time())[0]),
    }

    # Choose queue based on priority
    queue_key = KEY_PRIORITY_QUEUE if priority else KEY_WORK_QUEUE
    await redis_client.client.rpush(queue_key, json.dumps(task_data))
    logger.debug(f"Task {task_id} added to {'priority' if priority else 'work'} queue")


async def get_gemini_result(task_id: str, timeout: int = 600) -> Dict:
    # polls Redis for a task result until it appears or times out
    logger.debug(f"Waiting for result for task {task_id} with timeout {timeout}s")
    start_time = asyncio.get_event_loop().time()
    while True:
        result_json = await redis_client.client.hget(KEY_RESULTS_HASH, task_id)
        if result_json:
            await redis_client.client.hdel(KEY_RESULTS_HASH, task_id)
            response_data = json.loads(result_json)
            logger.debug(f"Retrieved result for task {task_id}")

            class MockUsage:
                def __init__(self, data):
                    data = data or {}
                    self.prompt_token_count = data.get("prompt_token_count", 0)
                    self.candidates_token_count = data.get("candidates_token_count", 0)
                    self.cached_content_token_count = data.get(
                        "cached_content_token_count", 0
                    )
                    self.thoughts_token_count = data.get("thoughts_token_count", 0)

            class MockResponse:
                def __init__(self, data):
                    self.text = data.get("text")
                    self.parsed = data.get("parsed")
                    self.usage_metadata = MockUsage(data.get("usage_metadata"))

            return MockResponse(response_data)

        if (asyncio.get_event_loop().time() - start_time) > timeout:
            logger.error(f"Timeout waiting for result for task {task_id}")
            raise asyncio.TimeoutError(f"Timeout waiting for result for task {task_id}")

        await asyncio.sleep(1)


async def get_gemini_result_with_retry(
    task_id: str, max_retries: int = 3, timeout: int = 600
) -> Dict:
    """Get Gemini result with retry logic and exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await get_gemini_result(task_id, timeout)
        except asyncio.TimeoutError as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Final timeout after {max_retries} attempts for task {task_id}"
                )
                raise e
            else:
                wait_time = 3**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Timeout on attempt {attempt + 1} for task {task_id}, retrying in {wait_time}s"
                )
                await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Final error after {max_retries} attempts for task {task_id}: {e}"
                )
                raise e
            else:
                wait_time = 3**attempt
                logger.warning(
                    f"Error on attempt {attempt + 1} for task {task_id}: {e}, retrying in {wait_time}s"
                )
                await asyncio.sleep(wait_time)


async def resubmit_task_with_priority(task_id: str, original_task_data: dict) -> None:
    """Resubmit a failed task to the priority queue"""
    original_task_data["retry_count"] = original_task_data.get("retry_count", 0) + 1
    original_task_data["retry_timestamp"] = int((await redis_client.client.time())[0])

    await redis_client.client.lpush(KEY_PRIORITY_QUEUE, json.dumps(original_task_data))
    logger.debug(
        f"Task {task_id} resubmitted to priority queue (retry #{original_task_data['retry_count']})"
    )


class GeminiWorker:
    def __init__(self):
        self.redis = redis_client.client
        self.initialized = False

    async def _ensure_rate_limits_initialized(self):
        """initialize rate limits if not already set"""
        if self.initialized:
            return

        pipe = self.redis.pipeline()
        for model_name, limits in MODEL_LIMITS.items():
            key_tpm = f"{KEY_TPM_PREFIX}:{model_name}"
            key_rpm = f"{KEY_RPM_PREFIX}:{model_name}"

            # only set if key doesn't exist
            pipe.setnx(key_tpm, limits["tpm"])
            pipe.setnx(key_rpm, limits["rpm"])

        await pipe.execute()
        self.initialized = True

    async def run(self):
        logger.info(
            "Gemini worker started. Processing jobs from priority and work queues."
        )
        await self._ensure_rate_limits_initialized()
        while True:
            # Check priority queue first, then regular queue
            task_json = None
            try:
                # Try priority queue first (non-blocking)
                task_json = await redis_client.client.rpop(KEY_PRIORITY_QUEUE)
                if not task_json:
                    # Try regular queue (blocking with timeout)
                    result = await redis_client.client.brpop(KEY_WORK_QUEUE, timeout=1)
                    if result:
                        _, task_json = result

                if not task_json:
                    await asyncio.sleep(0.1)  # Brief pause if no tasks
                    continue

            except Exception as e:
                logger.error(f"Error getting task from queue: {e}")
                await asyncio.sleep(1)
                continue

            task_data = json.loads(task_json)

            payload = task_data["payload"]
            task_id = task_data["task_id"]
            model_name = payload.get("model")
            logger.debug(f"Processing task {task_id} with model {model_name}")

            # dynamically construct the keys for this specific model
            key_tpm = f"{KEY_TPM_PREFIX}:{model_name}"
            key_rpm = f"{KEY_RPM_PREFIX}:{model_name}"

            lua_script = """
            local window_seconds = tonumber(ARGV[2])
            
            -- Get current request count (RPM)
            local requests_avail = tonumber(redis.call('get', KEYS[2]))
            
            -- Initialize if key doesn't exist
            if requests_avail == nil then
                requests_avail = tonumber(ARGV[1])
                redis.call('set', KEYS[2], ARGV[1])
            end
            
            -- Check if we have available requests (RPM check only)
            -- TPM will be checked and deducted AFTER actual token usage is known
            if requests_avail > 0 then
                -- Deduct one request
                redis.call('decr', KEYS[2])
                
                -- Set expiration for automatic reset (safety net)
                redis.call('expire', KEYS[2], window_seconds)
                
                return 1
            else
                return 0
            end
            """
            estimated_tokens = payload.get("estimated_tokens", 2500)

            model_limits = MODEL_LIMITS.get(
                model_name, {"tpm": 1000000, "rpm": 1000, "window_seconds": 60}
            )
            is_allowed = await self.redis.eval(
                lua_script,
                2,
                key_tpm,
                key_rpm,
                model_limits["rpm"],
                model_limits["window_seconds"],
            )

            if not is_allowed:
                logger.debug(
                    f"Rate limit exceeded for task {task_id}, requeuing to priority queue"
                )
                # Add retry count to task data
                task_data["retry_count"] = task_data.get("retry_count", 0) + 1
                task_data["retry_timestamp"] = int(
                    (await redis_client.client.time())[0]
                )

                # Use priority queue for retries to get faster processing
                await redis_client.client.lpush(
                    KEY_PRIORITY_QUEUE, json.dumps(task_data)
                )

                # Wait a bit longer for retries to avoid rapid requeuing
                await asyncio.sleep(0.5)
                continue

            # create background task without awaiting
            asyncio.create_task(
                self._process_gemini_task(
                    task_id, task_data, payload, key_tpm, model_name
                )
            )

    async def _process_gemini_task(
        self,
        task_id: str,
        task_data: Dict,
        payload: Dict,
        key_tpm: str,
        model_name: str,
    ):
        uploaded_file_name = None
        current_usage = {"prompt": 0, "output": 0, "thinking": 0, "cached": 0}

        try:
            logger.debug(f"Processing Gemini task {task_id}")

            # check if we need to process file content
            contents = payload["contents"]
            if payload.get("file_processing"):
                contents = []
                file_path = payload["file_path"]
                mime_type = payload.get("mime_type", "application/pdf")

                if payload["file_processing"] == "upload":
                    uploaded_file = await upload_policy_file(
                        pdf_path=file_path,
                        mime_type=mime_type,
                    )
                    contents.append(uploaded_file)
                    uploaded_file_name = uploaded_file.name

                elif payload["file_processing"] == "bytes":
                    file_bytes = await pdf_to_bytes(pdf_path=file_path)
                    contents.append(
                        types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                    )

                # add text content if exists
                if payload.get("text_contents"):
                    text_contents = payload["text_contents"]
                    for text in text_contents:
                        contents.append(types.Part.from_text(text=text))

            config_obj = GenerateContentConfig(**payload.get("config", {}))
            response = await gemini_client.aio.models.generate_content(
                model=payload["model"],
                contents=contents,
                config=config_obj,
            )

            # print("--------------------------------")
            # print("--------------------------------")
            # print(f"Response: {response}")
            # print("--------------------------------")
            # print("--------------------------------")

            if uploaded_file_name:
                try:
                    asyncio.create_task(
                        delete_policy_file_from_gemini_future(uploaded_file_name)
                    )
                except:
                    pass

            if response and response.usage_metadata:
                usage_info = response.usage_metadata
                actual_input_tokens = usage_info.prompt_token_count or 0
                actual_output_tokens = usage_info.candidates_token_count or 0
                actual_thinking_tokens = usage_info.thoughts_token_count or 0
                actual_cached_tokens = usage_info.cached_content_token_count or 0
                
                total_actual_tokens = (
                    actual_input_tokens
                    + actual_output_tokens
                    + actual_thinking_tokens
                    - actual_cached_tokens
                )

                current_usage["prompt"] = actual_input_tokens
                current_usage["output"] = actual_output_tokens
                current_usage["thinking"] = actual_thinking_tokens
                current_usage["cached"] = actual_cached_tokens

                model_limits = MODEL_LIMITS.get(
                    model_name, {"tpm": 1000000, "rpm": 1000, "window_seconds": 60}
                )
                
                lua_deduct_script = """
                local tokens_avail = tonumber(redis.call('get', KEYS[1]))
                
                if tokens_avail == nil then
                    tokens_avail = tonumber(ARGV[1])
                    redis.call('set', KEYS[1], ARGV[1])
                end
                
                -- Always deduct actual tokens used
                local tokens_to_deduct = tonumber(ARGV[2])
                redis.call('decrby', KEYS[1], tokens_to_deduct)
                redis.call('expire', KEYS[1], tonumber(ARGV[3]))
                
                return 1
                """
                
                await self.redis.eval(
                    lua_deduct_script,
                    1,
                    key_tpm,
                    model_limits["tpm"],
                    total_actual_tokens,
                    model_limits["window_seconds"],
                )
                
                logger.debug(
                    f"Task {task_id} consumed {total_actual_tokens} actual tokens "
                    f"(prompt: {actual_input_tokens}, output: {actual_output_tokens}, "
                    f"thinking: {actual_thinking_tokens}, cached: {actual_cached_tokens})"
                )

            # get accumulated tokens from previous retries
            previous_prompt = payload.get("accumulated_prompt_tokens", 0)
            previous_output = payload.get("accumulated_output_tokens", 0)
            previous_thinking = payload.get("accumulated_thinking_tokens", 0)
            previous_cached = payload.get("accumulated_cached_tokens", 0)

            response_payload = {
                "text": response.text,
                "parsed": response.parsed,
                "usage_metadata": {
                    "prompt_token_count": previous_prompt + current_usage["prompt"],
                    "candidates_token_count": previous_output + current_usage["output"],
                    "cached_content_token_count": previous_cached + current_usage["cached"],
                    "thoughts_token_count": previous_thinking
                    + current_usage["thinking"],
                },
            }
            await redis_client.client.hset(
                KEY_RESULTS_HASH, task_id, json.dumps(response_payload)
            )
            logger.debug(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Error processing Gemini task {task_id}: {e}", exc_info=True)

            # update payload with tokens consumed in this failed attempt
            payload["accumulated_prompt_tokens"] = (
                payload.get("accumulated_prompt_tokens", 0) + current_usage["prompt"]
            )
            payload["accumulated_output_tokens"] = (
                payload.get("accumulated_output_tokens", 0) + current_usage["output"]
            )
            payload["accumulated_thinking_tokens"] = (
                payload.get("accumulated_thinking_tokens", 0)
                + current_usage["thinking"]
            )
            payload["accumulated_cached_tokens"] = (
                payload.get("accumulated_cached_tokens", 0) + current_usage["cached"]
            )

            # update task_data with modified payload
            task_data["payload"] = payload

            # resubmit with accumulated tokens
            await resubmit_task_with_priority(task_id, task_data)


class RateLimitManager:
    """Background task to manage rate limits with proper sliding window"""

    async def run(self):
        logger.info("Rate Limit Manager started with sliding window management.")
        while True:
            await asyncio.sleep(
                30
            )  # Check every 30 seconds for more responsive management

            try:
                for model_name, limits in MODEL_LIMITS.items():
                    key_tpm = f"{KEY_TPM_PREFIX}:{model_name}"
                    key_rpm = f"{KEY_RPM_PREFIX}:{model_name}"

                    # Check if keys exist and have proper expiration
                    tpm_exists = await redis_client.client.exists(key_tpm)
                    rpm_exists = await redis_client.client.exists(key_rpm)

                    # If keys don't exist or are about to expire, reset them
                    if not tpm_exists:
                        await redis_client.client.setex(
                            key_tpm, limits["window_seconds"], limits["tpm"]
                        )
                        logger.debug(f"Reset TPM for {model_name}")

                    if not rpm_exists:
                        await redis_client.client.setex(
                            key_rpm, limits["window_seconds"], limits["rpm"]
                        )
                        logger.debug(f"Reset RPM for {model_name}")

                    # Check if we're close to expiration and need to refresh
                    tpm_ttl = await redis_client.client.ttl(key_tpm)
                    rpm_ttl = await redis_client.client.ttl(key_rpm)

                    if tpm_ttl < 10:  # Less than 10 seconds left
                        await redis_client.client.setex(
                            key_tpm, limits["window_seconds"], limits["tpm"]
                        )
                        logger.debug(f"Refreshed TPM for {model_name}")

                    if rpm_ttl < 10:  # Less than 10 seconds left
                        await redis_client.client.setex(
                            key_rpm, limits["window_seconds"], limits["rpm"]
                        )
                        logger.debug(f"Refreshed RPM for {model_name}")

            except Exception as e:
                logger.error(f"Error managing rate limits: {e}", exc_info=True)


async def clear_gemini_queues():
    # clear all gemini-related queues and results
    try:
        logger.info("Clearing Gemini queues and rate limit keys")
        pipe = redis_client.client.pipeline()
        pipe.delete(KEY_WORK_QUEUE)
        pipe.delete(KEY_PRIORITY_QUEUE)
        pipe.delete(KEY_RESULTS_HASH)
        # clear rate limit keys for all models
        for model_name in MODEL_LIMITS.keys():
            key_tpm = f"{KEY_TPM_PREFIX}:{model_name}"
            key_rpm = f"{KEY_RPM_PREFIX}:{model_name}"
            pipe.delete(key_tpm)
            pipe.delete(key_rpm)
        await pipe.execute()
        logger.info("All queues and rate limit keys cleared")
    except Exception as e:
        logger.error(f"Error clearing queues: {e}", exc_info=True)
