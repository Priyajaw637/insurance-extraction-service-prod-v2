import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    status,
)

from app.insurance_policy_processor.policy_orchestrator.agents.supervisor import (
    process_insurance_document,
)
from app.modules.async_redis import redis_client
from app.modules.Celery.tasks import (
    process_insurance_documents_commercial,
    process_proposal_documents_commercial,
    process_proposal_documents_commercial_us,
)
from app.utils.error_utils import error_handler
from app.logging_config import get_logger

logger = get_logger(__name__)

from app.insurance_policy_processor.policy_orchestrator.enums import (
    us_personal_extraction_instructions,
    commercial_first_level_instruction,
    commercial_second_level_instruction,
    commercial_extraction_instructions,
    us_personal_second_level_instruction,
)

insurance_commercial = APIRouter()


async def process_single_document(
    tool_name: str,
    data_mapping: Dict[str, Any],
    document_type: str,
    line_of_business: str,
    country: str,
    tool_process_id: str,
    commercial_extraction_schema: Dict[str, Any],
    tmp_dir: str = "/tmp",
    total_steps: int = 0,
    lob_possibilities: List[str] = [],
):
    try:
        first_level_instruction = ""
        second_level_instruction = ""
        extraction_instruction = ""

        async def update_progress(state):
            # Increment progress counter and publish update using redis_manager
            progress_percent = await redis_client.increment_progress_counter(
                task_id=tool_process_id, total_steps=total_steps
            )

            # Use safe progress update to prevent redundant messages
            await redis_client.publish_progress_update_safe(
                task_id=tool_process_id,
                state=state,
                progress=progress_percent,
                additional_data={"tool_process_id": tool_process_id},
            )

        await update_progress(f"{document_type.capitalize()} processing started")

        async def download_from_s3(s3_url: str, destination: str):
            async with aiohttp.ClientSession() as session:
                async with session.get(s3_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(destination, "wb") as f:
                            await f.write(await response.read())
                    else:
                        raise ValueError(
                            f"Failed to download {document_type} from S3: {response.status}"
                        )

        section = data_mapping.get(document_type)
        document_id = section.get("document_id")
        file_path = os.path.join(tmp_dir, f"{str(document_id)}.pdf")

        await download_from_s3(section["s3_link"], file_path)

        # Handle LOB filtering if needed
        logger.info(f"lob_possibilities: {lob_possibilities}")
        if lob_possibilities:
            if tool_name == "proposal_generation":
                lob_possibilities = ["insured_profile"] + list(lob_possibilities)
            if tool_name == "policy_checking" and country == "US":
                lob_possibilities = list(lob_possibilities) + ["general"]
            commercial_extraction_schema = {
                key: value
                for key, value in commercial_extraction_schema.items()
                if key in lob_possibilities
            }
        logger.info(f"lob_possibilities_list: {lob_possibilities}")
        logger.info(f"commercial_extraction_schema: {list(commercial_extraction_schema.keys())}")

        if line_of_business == "Personal" and country == "US":
            extraction_instruction = us_personal_extraction_instructions
            # first_level_instruction = us_personal_first_level_instruction
            second_level_instruction = us_personal_second_level_instruction
            # if isinstance(commercial_extraction_schema, dict) and len(commercial_extraction_schema) == 1:
            #     commercial_extraction_schema = next(iter(commercial_extraction_schema.values()))
        elif line_of_business == "Commercial" and country == "Canada":
            extraction_instruction = commercial_extraction_instructions
            first_level_instruction = commercial_first_level_instruction
            second_level_instruction = commercial_second_level_instruction

        logger.info(
            f"commercial_extraction_schema_after: {list(commercial_extraction_schema.keys())}"
        )

        asyncio.create_task(
            process_insurance_document(
                pdf_path=file_path,
                insurance_extraction_schema=commercial_extraction_schema,
                tmp_dir=tmp_dir,
                policy_document_id=document_id,
                list_lobs=lob_possibilities,
                line_of_business=line_of_business,
                country=country,
                tool_name=tool_name,
            )
        )

        # Create a separate Redis connection for this document to avoid conflicts
        from app.config import ENV_PROJECT
        from app.modules.async_redis import AsyncRedis

        document_redis = AsyncRedis(
            host=ENV_PROJECT.REDIS_URL, port=ENV_PROJECT.REDIS_PORT, db=1
        )

        # Subscribe to Redis channel for completion notification
        channel_name = f"insurance_document_processing_complete:{document_id}"
        await document_redis.subscribe(channel_name)

        # Wait indefinitely for a completion or error message
        while True:
            try:
                message = await document_redis.get_message(
                    timeout=60.0
                )  # Use a reasonable timeout for each check to allow handling interruptions
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        if data.get("policy_document_id") == document_id:
                            if data.get("status") == "completed":
                                await update_progress(
                                    f"{document_type.capitalize()} Processing Completed"
                                )
                                await document_redis.unsubscribe(channel_name)
                                await (
                                    document_redis.close()
                                )  # Close the separate Redis connection

                                os.makedirs(
                                    f"mapping/{tool_process_id}", exist_ok=True
                                )  # creates folder if it doesn't exist
                                with open(
                                    f"mapping/{tool_process_id}/{document_id}.json", "w"
                                ) as f:
                                    json.dump(data.get("state_data", {}), f, indent=4)
                                return {
                                    "success": True,
                                    "policy_document_id": document_id,
                                    "data": data.get("data", {}),
                                    "token_usage": data.get("token_usage", 0),
                                }
                            elif data.get("status") == "error":
                                await document_redis.unsubscribe(channel_name)
                                await (
                                    document_redis.close()
                                )  # Close the separate Redis connection
                                raise HTTPException(
                                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Error processing document: {data.get('error', 'Unknown error')}",
                                )
                    except json.JSONDecodeError:
                        continue  # Ignore invalid JSON messages
            except Exception as e:
                await document_redis.unsubscribe(channel_name)
                await document_redis.close()  # Close the separate Redis connection
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error while waiting for Redis message: {str(e)}",
                )

    except HTTPException:
        raise
    except Exception as e:
        redis_client.publish_task_failure(
            task_id=tool_process_id,
            error_message=str(e),
            additional_data={"tool_process_id": tool_process_id},
        )
        raise
    finally:
        # Ensure Redis subscription is cleaned up
        try:
            if "document_redis" in locals():
                await document_redis.unsubscribe(channel_name)
                await document_redis.close()
        except:
            pass  # Ignore errors during cleanup


@insurance_commercial.post("/insurance/extract_data")
async def extract_insurance_data_commercial(
    policy_comparision_id: str = Form(...),
    data_mapping: Optional[str] = Form(None),
    commercial_extraction_schema: Optional[str] = Form(None),
    lob_possibilities: Optional[str] = Form(None),
    line_of_business: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
):
    try:
        await redis_client.publish_task_update(
            task_id=policy_comparision_id,
            state="Policy Checking Started",
            progress=15,
        )
        results = process_insurance_documents_commercial.delay(
            data_mapping=data_mapping,
            policy_comparision_id=policy_comparision_id,
            line_of_business=line_of_business,
            commercial_extraction_schema=commercial_extraction_schema,
            lob_possibilities=lob_possibilities,
            country=country,
        )
        if results is None:
            raise error_handler.create_http_error_response(
                500, "Document processing failed"
            )

        return {"task_id": policy_comparision_id}

    except HTTPException:
        raise
    except Exception as e:
        await error_handler.handle_task_error(
            task_id=policy_comparision_id, error=e, context="Commercial policy checking"
        )
        raise error_handler.create_http_error_response(
            500, f"Internal server error: {str(e)}"
        )


@insurance_commercial.post("/proposal/extract_data")
async def extract_proposal_data_commercial(
    proposal_id: Optional[str] = Form(...),
    data_format: Optional[str] = Form(None),
    commercial_extraction_schema: Optional[str] = Form(None),
    lob_possibilities: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    line_of_business: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
):
    try:
        await redis_client.publish_task_update(
            task_id=proposal_id,
            state="Proposal Generation Started",
            progress=15,
        )

        if country == "US" and line_of_business in {"Personal", "Commercial"}:
            results = process_proposal_documents_commercial_us.delay(
                proposal_id=proposal_id,
                data_format=data_format,
                commercial_extraction_schema=commercial_extraction_schema,
                context=context,
                lob_possibilities=lob_possibilities,
                line_of_business=line_of_business,
                country=country,
            )

        else:
            results = process_proposal_documents_commercial.delay(
                proposal_id=proposal_id,
                data_format=data_format,
                commercial_extraction_schema=commercial_extraction_schema,
                context=context,
                lob_possibilities=lob_possibilities,
                line_of_business=line_of_business,
                country=country,
            )

        if results is None:
            raise error_handler.create_http_error_response(
                500, "Document processing failed"
            )

        return {"task_id": proposal_id}

    except HTTPException:
        raise
    except Exception as e:
        await error_handler.handle_task_error(
            task_id=proposal_id, error=e, context="Commercial proposal generation"
        )
        raise error_handler.create_http_error_response(
            500, f"Internal server error: {str(e)}"
        )
