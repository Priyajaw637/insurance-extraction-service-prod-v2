import asyncio

from celery.signals import task_failure

from app.modules.async_redis import redis_client
from app.utils.logger_factory import get_logger

logger = get_logger(__name__)


@task_failure.connect
def task_failed_handler(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **extras,
):
    """Optimized task failure handler with cleaner async handling."""

    async def handle_failure():
        return await async_task_failed_handler(
            sender=sender,
            task_id=task_id,
            exception=exception,
            args=args,
            kwargs=kwargs,
            traceback=traceback,
            einfo=einfo,
            **extras,
        )

    try:
        # Use asyncio.run() for cleaner event loop management
        asyncio.run(handle_failure())
    except Exception as e:
        # Log error but don't raise to avoid breaking Celery
        logger.error(f"Error in task failure handler: {e}", exc_info=True)


async def async_task_failed_handler(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **extras,
):
    """
    Async handler for task failures using centralized Redis utilities
    """
    error_message = str(exception)
    logger.error(f"Task {task_id} failed with exception: {exception}")

    # Debug: Check what kwargs actually is
    logger.debug(f"kwargs type: {type(kwargs)}")
    logger.debug(f"kwargs value: {kwargs}")

    # Safely handle kwargs - ensure it's a dictionary
    if not isinstance(kwargs, dict):
        logger.warning(f"kwargs is not a dict, it's {type(kwargs)}. Using empty dict.")
        kwargs = {}

    # Extract identifiers from kwargs safely
    proposal_id = kwargs.get("proposal_id")
    policy_comparision_id = kwargs.get("policy_comparision_id")

    logger.debug(f"proposal_id: {proposal_id}")
    logger.debug(f"policy_comparision_id: {policy_comparision_id}")

    try:
        if proposal_id:
            await redis_client.publish_task_failure(
                task_id=proposal_id,
                error_message=error_message,
                additional_data={"proposal_id": proposal_id},
            )

        if policy_comparision_id:
            logger.info("Publishing policy comparison failure")
            await redis_client.publish_task_failure(
                task_id=policy_comparision_id,
                error_message=error_message,
                additional_data={"policy_comparision_id": policy_comparision_id},
            )

        if not proposal_id and not policy_comparision_id:
            logger.warning(f"No ID found in task kwargs: {kwargs}")

    except Exception as redis_err:
        print(f"Redis publish error: {redis_err}")
