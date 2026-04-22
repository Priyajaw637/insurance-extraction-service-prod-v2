import json
from typing import Any, Dict, Optional

import redis.asyncio as redis

from app.config import ENV_PROJECT
from app.logging_config import get_logger


logger = get_logger(__name__)


class AsyncRedis:
    def __init__(self, host: str, port: int, db: int, **kwargs):
        self.client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True, **kwargs
        )
        self.pubsub = self.client.pubsub()
        self.pubsub.ignore_subscribe_messages = True
        self.threads_workers = {}
        self.email_workers = {}

    async def create(self, key: str, value: Any, expire: int = 14400):
        try:
            value = json.dumps(value)
            result = await self.client.set(key, value, ex=expire)
            return result
        except Exception as e:
            logger.error(f"Redis Create Error: {e}", exc_info=True)
            return None

    async def get_by_key(self, key: str):
        if not key:
            return None
        value = await self.client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception as e:
            return value

    async def lists(self, pattern: str = "*"):
        return await self.client.keys(pattern)

    async def update(self, key: str, value: Any, expire: int = 14400):
        if not key:
            return False
        value = json.dumps(value)
        result = await self.client.set(key, value, ex=expire)
        return result

    async def delete(self, key: str):
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            pass

    async def exists(self, key: str):
        return await self.client.exists(key) > 0

    async def publish(self, channel: str, message: Any):
        """Publish a message to a Redis channel"""
        try:
            message = json.dumps(message)
            result = await self.client.publish(channel, message)
            return result
        except Exception as e:
            print(f"Redis Publish Error: {e}")
            return None

    async def subscribe(self, channel: str):
        """Subscribe to a Redis channel and return the pubsub object"""
        try:
            await self.pubsub.subscribe(channel)
            return self.pubsub
        except Exception as e:
            print(f"Redis Subscribe Error: {e}")
            return None

    async def unsubscribe(self, channel: str):
        """Unsubscribe from a Redis channel"""
        try:
            await self.pubsub.unsubscribe(channel)
            return True
        except Exception as e:
            print(f"Redis Unsubscribe Error: {e}")
            return False

    async def get_message(self, timeout: float = 1.0):
        """Get a message from the subscribed channel with timeout"""
        try:
            message = await self.pubsub.get_message(timeout=timeout)
            return message
        except Exception as e:
            print(f"Redis Get Message Error: {e}")
            return None

    async def close(self):
        await self.pubsub.close()
        await self.client.aclose()

    async def publish_task_update(
        self,
        task_id: str,
        state: str,
        progress: int,
        error: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish task update to Redis channel and set latest update.

        Args:
            task_id: Task identifier
            state: Current task state
            progress: Progress percentage (0-100)
            error: Error message if any
            additional_data: Additional data to include in update
        """
        try:
            update_data = {"state": state, "progress": progress, "task_id": task_id}

            if error:
                update_data["error"] = error

            if additional_data:
                update_data.update(additional_data)

            update_json = json.dumps(update_data)

            pipe = self.client.pipeline()
            pipe.publish(f"task_updates:{task_id}", update_json)
            pipe.set(f"latest_tasks_update_{task_id}", update_json)
            await pipe.execute()

        except Exception as e:
            logger.error(f"Failed to publish task update for {task_id}: {e}")

    async def publish_task_failure(
        self,
        task_id: str,
        error_message: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish task failure notification with error details."""
        try:
            failure_data = {
                "state": "Failed",
                "progress": -1,
                "error": error_message,
                "task_id": task_id,
            }

            if additional_data:
                failure_data.update(additional_data)

            await self.publish_task_update(
                task_id=task_id,
                state="Failed",
                progress=-1,
                error=error_message,
                additional_data=additional_data,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish task failure for {task_id}: {e}")
            return False

    async def increment_progress_counter(self, task_id: str, total_steps: int) -> int:
        """Increment progress counter and return progress percentage (max 90%)."""
        try:
            progress_key = f"task_progress_counter:{task_id}"
            completion_key = f"task_completed:{task_id}"

            # Check if task is already completed
            is_completed = await self.client.get(completion_key)
            if is_completed:
                return 90  # Cap progress from this stage at 90%

            # Initialize counter if it doesn't exist
            await self.client.setnx(progress_key, 0)

            # Increment and get the new value
            completed_steps = await self.client.incr(progress_key)

            # Compute capped progress percentage
            progress_percent = int((completed_steps / total_steps) * 90)
            if progress_percent > 90:
                progress_percent = 90

            # Optional: Mark as "stage complete" if 90% reached (not full completion)
            if progress_percent >= 90:
                await self.client.set(
                    completion_key, "stage_complete", ex=3600
                )  # expire after 1 hour

            return progress_percent

        except Exception as e:
            logger.error(f"Failed to increment progress counter for {task_id}: {e}")
            return 0

    async def publish_progress_update_safe(
        self,
        task_id: str,
        state: str,
        progress: int,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Safely publish progress update, preventing duplicate updates after completion."""
        try:
            completion_key = f"task_completed:{task_id}"

            # Check if task is already completed
            is_completed = await self.client.get(completion_key)
            if is_completed and progress < 100:
                logger.info(
                    f"Skipping progress update for completed task {task_id}: {state}"
                )
                return False

            # Publish the update
            await self.publish_task_update(
                task_id, state, progress, additional_data=additional_data
            )
            return True

        except Exception as e:
            logger.error(f"Failed to safely publish progress update for {task_id}: {e}")
            return False

    async def mark_task_completed(
        self, task_id: str, expire_seconds: int = 3600
    ) -> bool:
        """Mark a task as completed with optional expiration."""
        try:
            completion_key = f"task_completed:{task_id}"
            await self.client.set(completion_key, "1", ex=expire_seconds)
            return True
        except Exception as e:
            logger.error(f"Failed to mark task {task_id} as completed: {e}")
            return False


redis_client = AsyncRedis(host=ENV_PROJECT.REDIS_URL, port=ENV_PROJECT.REDIS_PORT, db=1)
