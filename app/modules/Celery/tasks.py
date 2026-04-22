import asyncio
import json
import os
import re
from datetime import datetime
import aiohttp

from app.modules.async_redis import redis_client
from .config import celery
from app.policy_tools.commercial_comparison import commercial_compare_jsons
from app.policy_tools.policy_data_flattener import flatten_json_with_values
from app.policy_tools.carrier_recommendation import recommend_best_carrier
from app.utils.logger_factory import get_logger, set_task_id, clear_task_id
from app.config import ENV_PROJECT

logger = get_logger(__name__)


import threading

_loop_local = threading.local()

def get_or_create_loop():
    """Get existing event loop for this thread or create a new persistent one."""
    if not hasattr(_loop_local, 'loop') or _loop_local.loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop_local.loop = loop
    return _loop_local.loop

def run_async_task_safely(coro):
    loop = get_or_create_loop()
    return loop.run_until_complete(coro)


@celery.task(bind=True)
def process_insurance_documents_commercial(
    self,
    data_mapping: str,
    policy_comparision_id: str = None,
    line_of_business: str = None,
    country: str = None,
    package_id: str = None,
    commercial_extraction_schema: str = None,
    lob_possibilities: str = None,
    **kwargs,
):
    from app.routes.insurance_commercial import process_single_document
    # dump_data = {
    #     "timestamp": datetime.utcnow().isoformat(),
    #     "policy_comparision_id": policy_comparision_id,
    #     "line_of_business": line_of_business,
    #     "country": country,
    #     "package_id": package_id,
    #     "lob_possibilities_raw": lob_possibilities,
    #     "commercial_extraction_schema_raw": commercial_extraction_schema,
    #     "data_mapping_raw": data_mapping,
    #     "extra_kwargs": kwargs,
    # }

    # with open("process_insurance_documents_commercial_input.json", "w") as f:
    #     json.dump(dump_data, f, indent=4)
    
    async def run_task():
        set_task_id(policy_comparision_id)
        logger.info(
            f"Starting extraction task for policy_comparision_id: {policy_comparision_id}"
        )
        data_mapping_dict = {}

        try:
            data_mapping_dict = json.loads(data_mapping) if data_mapping else {}
            commercial_extraction_schema_dict = (
                json.loads(commercial_extraction_schema)
                if commercial_extraction_schema
                else {}
            )
            lob_possibilities_list = (
                json.loads(lob_possibilities) if lob_possibilities else []
            )

            # ----------------------------
            # LOAD SCHEMA IF REQUIRED
            # ----------------------------
            # if line_of_business == "Personal" and country == "US":
            #     with open("US_personal_policy_schema.json", "r") as f:
            #         commercial_extraction_schema_dict = json.load(f)

            # ----------------------------
            # SPLIT DOCUMENTS
            # ----------------------------
            to_extract = {}
            pre_extracted = {}

            for doc_type, value in data_mapping_dict.items():
                if not value.get("document_id"):
                    continue

                if country == "US" and line_of_business in {"Personal", "Commercial"}:
                    if value.get("commercial_extracted_data_us"):
                        pre_extracted[doc_type] = value
                    else:
                        to_extract[doc_type] = value
                else:
                    if value.get("commercial_extracted_data"):
                        pre_extracted[doc_type] = value
                    else:
                        to_extract[doc_type] = value

            total_steps = len(to_extract) * 2

            # ----------------------------
            # REUSE PRE-EXTRACTED DATA
            # ----------------------------
            unique_first_level_keys = set()

            for doc_type, value in pre_extracted.items():
                if country == "US" and line_of_business in {"Personal", "Commercial"}:
                    extracted_data = value.get("commercial_extracted_data_us", {})
                else:
                    extracted_data = value.get("commercial_extracted_data", {})

                unique_first_level_keys.update(
                    key
                    for key, val in extracted_data.items()
                    if val
                    and key
                    not in {"token_consumptions", "insured_profile", "program_pricing_matrix","general"}
                )

            # ----------------------------
            # PROCESS REQUIRED DOCS
            # ----------------------------
            async def process_doc(doc_type):
                result = await process_single_document(
                    tool_name="policy_checking",
                    data_mapping=data_mapping_dict,
                    document_type=doc_type,
                    line_of_business=line_of_business,
                    country=country,
                    tool_process_id=policy_comparision_id,
                    commercial_extraction_schema=commercial_extraction_schema_dict,
                    total_steps=total_steps,
                    lob_possibilities=lob_possibilities_list,
                )
                return doc_type, result["data"], result["token_usage"]

            tasks = [process_doc(k) for k in to_extract]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_docs = 0
            failed_docs = 0

            for res in results:
                if isinstance(res, Exception):
                    failed_docs += 1
                    logger.error(res, exc_info=True)
                    continue

                successful_docs += 1
                doc_type, extracted_data, token_usage = res

                if country == "US" and line_of_business in {"Personal", "Commercial"}:
                    data_mapping_dict[doc_type].setdefault(
                        "commercial_extracted_data_us", {}
                    )
                    data_mapping_dict[doc_type]["commercial_extracted_data_us"] = extracted_data
                else:
                    data_mapping_dict[doc_type].setdefault(
                        "commercial_extracted_data", {}
                    )
                    data_mapping_dict[doc_type]["commercial_extracted_data"] = extracted_data

                data_mapping_dict[doc_type]["token_usage"] = token_usage

                unique_first_level_keys.update(
                    key
                    for key, val in extracted_data.items()
                    if val
                    and key
                    not in {"token_consumptions", "insured_profile", "program_pricing_matrix","general"}
                )

            if failed_docs == len(to_extract) and to_extract:
                raise Exception("All document extraction tasks failed")

            # ----------------------------
            # COMPARE
            # ----------------------------
            extracted_field = (
                "commercial_extracted_data_us"
                if country == "US" and line_of_business in {"Personal", "Commercial"}
                else "commercial_extracted_data"
            )

            async def update_progress(state, progress):
                # Mark task as completed when reaching 100%
                if progress >= 100:
                    await redis_client.mark_task_completed(policy_comparision_id)

                await redis_client.publish_task_update(
                    task_id=policy_comparision_id,
                    state=state,
                    progress=progress,
                    additional_data={"policy_comparision_id": policy_comparision_id},
                )

            logger.info(f"Starting comparison | extracted_field={extracted_field} | docs={list(data_mapping_dict.keys())}")
            anomalies = await commercial_compare_jsons(
                policy=data_mapping_dict.get("Policy", {}).get(extracted_field),
                binder=data_mapping_dict.get("Binder", {}).get(extracted_field),
                quote=data_mapping_dict.get("Quote", {}).get(extracted_field),
            )
            logger.info(f"Comparison completed | total={anomalies['total_fields']} matched={anomalies['matched_count']} mismatched={anomalies['mismatched_count']}")

            logger.info("Storing policy checking results")
            async with aiohttp.ClientSession() as session:
                serialized_anomalies = await asyncio.to_thread(
                    lambda: [a.model_dump() for a in anomalies["anomalies"]]
                )

                store_resp = await session.post(
                    "http://127.0.0.1:8000/api/v1/user/store/commercialpolicychecking/data",
                    json={
                        "policy_comparision_id": policy_comparision_id,
                        "lob_type": list(unique_first_level_keys),
                        "data_mapping": data_mapping_dict,
                        "anomaly": serialized_anomalies,
                        "field_count": {
                            "matched_count": anomalies["matched_count"],
                            "mismatched_count": anomalies["mismatched_count"],
                            "approved_count": 0,
                            "total_fields": anomalies["total_fields"],
                        },
                        "gpt_verification_tokens": anomalies["token"],
                        "line_of_business": line_of_business,
                        "country": country,
                    },
                )
                logger.info(f"Store API response status: {store_resp.status}")

            await update_progress("Policy Checking Completed", 100)
            logger.info("Policy checking task completed successfully")

        except Exception as e:
            logger.error(f"Policy checking task FAILED: {str(e)}", exc_info=True)
            await redis_client.publish_task_failure(
                task_id=policy_comparision_id,
                error_message=str(e),
                additional_data={"policy_comparision_id": policy_comparision_id},
            )
            raise

        finally:
            cleanup_tasks = []
            for value in data_mapping_dict.values():
                file_path = f"/tmp/{str(value.get('document_id'))}.pdf"
                if await asyncio.to_thread(os.path.exists, file_path):
                    cleanup_tasks.append(asyncio.to_thread(os.remove, file_path))
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)
            clear_task_id(policy_comparision_id)

    async def run_task_with_timeout():
        try:
            return await asyncio.wait_for(run_task(), timeout=3600)  # 60 minutes timeout
        except asyncio.TimeoutError:
            logger.error(f"Task {policy_comparision_id} timed out after 1 hour")
            await redis_client.publish_task_failure(
                task_id=policy_comparision_id,
                error_message="Task completely stuck or timed out after 1 hour limit.",
                additional_data={"policy_comparision_id": policy_comparision_id},
            )
            raise Exception("Task timed out after 3600 seconds")

    return run_async_task_safely(run_task_with_timeout())


@celery.task(bind=True)
def process_proposal_documents_commercial(
    self,
    proposal_id: str,
    data_format: str,
    commercial_extraction_schema: str,
    context: str,
    lob_possibilities: str,
    line_of_business: str,
    country: str,
    **kwargs,
):
    from app.routes.insurance_commercial import process_single_document

    async def run_task():
        set_task_id(proposal_id)
        logger.info(f"Starting extraction task for proposal_id: {proposal_id}")
        data_mapping_dict = {}
        proposal_comparison_data = {}

        try:
            data_mapping_dict = json.loads(data_format) if data_format else {}
            commercial_extraction_schema_dict = (
                json.loads(commercial_extraction_schema)
                if commercial_extraction_schema
                else {}
            )
            lob_possibilities_list = (
                json.loads(lob_possibilities) if lob_possibilities else []
            )

            with open("commercial_proposal_schema.json", "r") as file:
                commercial_extraction_schema_dict = json.load(file)

            # ---- SPLIT DOCS ----
            to_extract = {}
            pre_extracted = {}

            for k, v in data_mapping_dict.items():
                if not v.get("document_id"):
                    continue
                if v.get("commercial_proposal_data"):
                    pre_extracted[k] = v
                else:
                    to_extract[k] = v

            total_steps = len(to_extract) * 2

            unique_first_level_keys = set()

            # ---- REUSE PRE-EXTRACTED ----
            for doc_type, value in pre_extracted.items():
                extracted_data = value["commercial_proposal_data"]
                data_mapping_dict[doc_type]["token_usage"] = {}

                # ✅ ADD UNIQUE FIRST LEVEL KEYS (PRE-UPLOADED)
                unique_first_level_keys.update(
                    k
                    for k, v in extracted_data.items()
                    if v
                    and k
                    not in {
                        "token_consumptions",
                        "insured_profile",
                        "program_pricing_matrix",
                        "summary_of_pricing",
                    }
                )

                carrier_name = None
                for lob_name, lob_details in extracted_data.items():
                    if isinstance(lob_details, dict):
                        insurer_name = (
                            lob_details.get("policy_identification", {})
                            .get("insurer_name")
                        )
                        if insurer_name:
                            carrier_name = insurer_name.strip()
                            break

                carrier_key = carrier_name or doc_type

                # --- Handle unique key if multiple docs have same carrier ---
                suffix = 1
                unique_key = carrier_key
                while unique_key in proposal_comparison_data:
                    suffix += 1
                    unique_key = f"{carrier_key} ({suffix})"

                proposal_comparison_data[unique_key] = extracted_data

            # ---- PROCESS ONLY REQUIRED DOCS ----
            async def process_doc(doc_type):
                result = await process_single_document(
                    tool_name="proposal_generation",
                    data_mapping=data_mapping_dict,
                    document_type=doc_type,
                    tool_process_id=proposal_id,
                    commercial_extraction_schema=commercial_extraction_schema_dict,
                    total_steps=total_steps,
                    lob_possibilities=lob_possibilities_list,
                    line_of_business=line_of_business,
                    country=country,
                )
                return doc_type, result["data"], result["token_usage"]

            tasks = [process_doc(k) for k in to_extract]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            def parse_amount(val):
                if not isinstance(val, str):
                    return 0.0
                cleaned = re.sub(r"[^\d.]", "", val)
                try:
                    return float(cleaned)
                except Exception:
                    return 0.0

            for res in results:
                if isinstance(res, Exception):
                    logger.error(res, exc_info=True)
                    continue

                doc_type, extracted_data, token_usage = res
                extracted_data = await flatten_json_with_values(extracted_data)

                data_mapping_dict[doc_type]["commercial_proposal_data"] = extracted_data
                data_mapping_dict[doc_type]["token_usage"] = token_usage

                carrier_name = None
                for lob_name, lob_details in extracted_data.items():
                    if isinstance(lob_details, dict):
                        insurer_name = (
                            lob_details.get("policy_identification", {})
                            .get("insurer_name")
                        )
                        if insurer_name:
                            carrier_name = insurer_name.strip()
                            break

                carrier_key = carrier_name or doc_type

                # --- Add to proposal comparison data ---
                suffix = 1
                unique_key = carrier_key
                while unique_key in proposal_comparison_data:
                    suffix += 1
                    unique_key = f"{carrier_key} ({suffix})"

                proposal_comparison_data[unique_key] = extracted_data

                base = terror = all_in = 0.0
                for _, v in extracted_data.items():
                    pricing = v.get("program_pricing_matrix", {})
                    base += parse_amount(pricing.get("base_premium_cad"))
                    terror += parse_amount(pricing.get("terrorism_premium_cad"))
                    all_in += parse_amount(pricing.get("all_in_premium_cad"))

                summary = {
                    "total_base_premium_cad": base,
                    "total_terrorism_premium_cad": terror,
                    "total_all_in_premium_cad": all_in,
                }

                data_mapping_dict[doc_type]["commercial_proposal_data"][
                    "summary_of_pricing"
                ] = summary
                proposal_comparison_data[carrier_key]["summary_of_pricing"] = summary

                # ✅ ADD UNIQUE FIRST LEVEL KEYS (EXTRACTED)
                unique_first_level_keys.update(
                    k
                    for k, v in extracted_data.items()
                    if v
                    and k
                    not in {
                        "token_consumptions",
                        "insured_profile",
                        "program_pricing_matrix",
                        "summary_of_pricing"
                    }
                )
            
            async def update_progress(state, progress):
                # Mark task as completed when reaching 100%
                if progress >= 100:
                    await redis_client.mark_task_completed(proposal_id)

                await redis_client.publish_task_update(
                    task_id=proposal_id,
                    state=state,
                    progress=progress,
                    additional_data={"proposal_id": proposal_id},
                )

            carriers_list = []
            for carrier_key, carrier_data in data_mapping_dict.items():
                lob_details = carrier_data.get("commercial_proposal_data", {})
                insurer_name = None
                for lob_key, lob_val in lob_details.items():
                    try:
                        name = lob_val.get("policy_identification", {}).get(
                            "insurer_name"
                        )
                        if name:
                            insurer_name = name
                            break
                    except Exception:
                        pass

                carriers_list.append(
                    {
                        "carrier_name": insurer_name or carrier_key,
                        "lob_details": lob_details,
                    }
                )

            formatted_data = {"carriers": carriers_list}
            try:
                logger.info("Generating carrier recommendation")
                recommendation = await recommend_best_carrier(formatted_data, context)
                proposal_comparison_data["recommendation"] = recommendation["result"]
                recommendation_tokens = recommendation.get("token", {})
                logger.info("Carrier recommendation completed")
            except Exception as e:
                logger.error(f"Carrier recommendation FAILED: {str(e)}", exc_info=True)
                proposal_comparison_data["recommendation"] = {
                    "recommended_carrier": "N/A",
                    "recommendation_text": "Error generating recommendation. Please review comparison manually."
                }
                recommendation_tokens = {}


            logger.info("Storing proposal generation results")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://127.0.0.1:8000/api/v1/user/store/commercialproposal/data",
                    json={
                        "proposal_id": proposal_id,
                        "lob_type": list(unique_first_level_keys),
                        "data_mapping": data_mapping_dict,
                        "proposal_comparison_data": proposal_comparison_data,
                        "recommendation_tokens": recommendation_tokens,
                        "line_of_business": line_of_business,
                        "country": country,
                    },
                ) as resp:
                    await resp.json()
                    logger.info(f"Store API response status: {resp.status}")

            await update_progress("Proposal Generation Completed", 100)
            logger.info("Proposal generation task completed successfully")

        except Exception as e:
            logger.error(f"Proposal generation task FAILED: {str(e)}", exc_info=True)
            await redis_client.publish_task_failure(
                task_id=proposal_id,
                error_message=str(e),
                additional_data={"proposal_id": proposal_id},
            )
            raise

        finally:
            cleanup_tasks = []
            for value in data_mapping_dict.values():
                file_path = f"/tmp/{str(value.get('document_id'))}.pdf"
                if await asyncio.to_thread(os.path.exists, file_path):
                    cleanup_tasks.append(asyncio.to_thread(os.remove, file_path))
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)
                logger.info("Temporary files deleted successfully.")
            clear_task_id(proposal_id)

    async def run_task_with_timeout():
        try:
            return await asyncio.wait_for(run_task(), timeout=3600)  # 60 minutes timeout
        except asyncio.TimeoutError:
            logger.error(f"Task {proposal_id} timed out after 1 hour")
            await redis_client.publish_task_failure(
                task_id=proposal_id,
                error_message="Task completely stuck or timed out after 1 hour limit.",
                additional_data={"proposal_id": proposal_id},
            )
            raise Exception("Task timed out after 3600 seconds")

    return run_async_task_safely(run_task_with_timeout())


@celery.task(bind=True)
def process_proposal_documents_commercial_us(
    self,
    proposal_id: str,
    data_format: str,
    commercial_extraction_schema: str,
    context: str,
    lob_possibilities: str,
    line_of_business: str,
    country: str,
    **kwargs,
):
    """Process commercial proposal documents with optimized async handling."""
    from app.routes.insurance_commercial import process_single_document

    # dump_data = {
    # "timestamp": datetime.utcnow().isoformat(),
    # "proposal_id": proposal_id,
    # "data_format": data_format,
    # "commercial_extraction_schema": commercial_extraction_schema,
    # "context": context,
    # "lob_possibilities": lob_possibilities,
    # "line_of_business": line_of_business,
    # "country": country,
    # "kwargs": kwargs,
    # }
    # with open("process_insurance_documents_commercial_input.json", "w") as f:
    #     json.dump(dump_data, f, indent=4)

    async def run_task():
        set_task_id(proposal_id)
        logger.info(f"Starting extraction task for proposal_id: {proposal_id}")
        data_mapping_dict = {}
        proposal_comparison_data = {}

        try:
            # Parse JSON strings
            data_mapping_dict = json.loads(data_format) if data_format else {}
            commercial_extraction_schema_dict = (
                json.loads(commercial_extraction_schema)
                if commercial_extraction_schema
                else {}
            )
            lob_possibilities_list = (
                json.loads(lob_possibilities) if lob_possibilities else []
            )
            # ---- SPLIT DOCS ----
            to_extract = {}
            pre_extracted = {}

            for k, v in data_mapping_dict.items():
                if not v.get("document_id"):
                    continue
                if v.get("commercial_proposal_data_us"):
                    pre_extracted[k] = v
                else:
                    to_extract[k] = v

            total_steps = len(to_extract) * 2

            unique_first_level_keys = set()
            with open("US_commercial_proposal_schema.json", "r") as file:
                commercial_extraction_schema_dict = json.load(file)


            def parse_amount(value: str) -> float:
                """Safely parse currency-like text into float, e.g. '$1,234.56' → 1234.56.
                Returns 0.0 for invalid or non-numeric values (like 'one thousand', 'N/A', etc.).
                """
                if not value or not isinstance(value, str):
                    return 0.0

                # Extract numeric part
                cleaned = re.sub(r"[^\d.]", "", value)

                # Handle empty or invalid cases (e.g., no digits found)
                if not cleaned or cleaned.count(".") > 1:
                    return 0.0

                try:
                    return float(cleaned)
                except ValueError:
                    return 0.0

            # ---- REUSE PRE-EXTRACTED ----
            for doc_type, value in pre_extracted.items():
                extracted_data = value["commercial_proposal_data_us"]
                data_mapping_dict[doc_type]["token_usage"] = {}

                # ✅ ADD UNIQUE FIRST LEVEL KEYS (PRE-UPLOADED)
                unique_first_level_keys.update(
                    k
                    for k, v in extracted_data.items()
                    if v
                    and k
                    not in {
                        "token_consumptions",
                        "insured_profile",
                        "program_pricing_matrix",
                        "summary_of_pricing",
                    }
                )

                carrier_name = None
                for lob_name, lob_details in extracted_data.items():
                    if isinstance(lob_details, dict):
                        insurer_name = (
                            lob_details.get("program_pricing_matrix", {})
                            .get("carrier")
                        )
                        if insurer_name:
                            carrier_name = insurer_name.strip()
                            break

                carrier_key = carrier_name or doc_type
                
                # --- Handle unique key if multiple docs have same carrier ---
                suffix = 1
                unique_key = carrier_key
                while unique_key in proposal_comparison_data:
                    suffix += 1
                    unique_key = f"{carrier_key} ({suffix})"

                proposal_comparison_data[unique_key] = extracted_data

            # Process documents concurrently
            async def process_doc(doc_type):
                result = await process_single_document(
                    tool_name="proposal_generation",
                    data_mapping=data_mapping_dict,
                    document_type=doc_type,
                    tool_process_id=proposal_id,
                    commercial_extraction_schema=commercial_extraction_schema_dict,
                    total_steps=total_steps,
                    lob_possibilities=lob_possibilities_list,
                    line_of_business=line_of_business,
                    country=country,
                )
                return doc_type, result["data"], result["token_usage"]

            # Execute document processing tasks concurrently
            doc_tasks = [process_doc(k) for k in to_extract]
            results = await asyncio.gather(*doc_tasks, return_exceptions=True)

            successful_docs = 0
            failed_docs = 0

            for result in results:
                if isinstance(result, Exception):
                    failed_docs += 1
                    logger.error(f"Error processing document: {result}", exc_info=True)
                    continue  # skip to next result

                successful_docs += 1
                doc_type, extracted_data, token_usage = result
                extracted_data = await flatten_json_with_values(extracted_data)
                data_mapping_dict[doc_type]["commercial_proposal_data_us"] = extracted_data
                data_mapping_dict[doc_type]["token_usage"] = token_usage

                # --- Extract actual carrier name ---
                carrier_name = None
                for lob_name, lob_details in extracted_data.items():
                    if isinstance(lob_details, dict):
                        insurer_name = (
                            lob_details.get("program_pricing_matrix", {}).get(
                                "carrier"
                            )
                            if lob_details.get("program_pricing_matrix")
                            else None
                        )
                        if insurer_name:
                            carrier_name = insurer_name.strip()
                            break

                # fallback to doc_type if no carrier name found
                carrier_key = carrier_name or doc_type

                # --- Add to proposal comparison data ---
                suffix = 1
                unique_key = carrier_key
                while unique_key in proposal_comparison_data:
                    suffix += 1
                    unique_key = f"{carrier_key} ({suffix})"

                proposal_comparison_data[unique_key] = extracted_data

                # Only include non-empty top-level keys
                unique_first_level_keys.update(
                    key
                    for key, value in extracted_data.items()
                    if value != {}
                    and key
                    not in {
                        "token_consumptions",
                        "insured_profile",
                        "program_pricing_matrix",
                        "summary_of_pricing",
                    }
                )

                # Reset totals per document
                total_base = total_terrorism = total_all_in = 0

                for key, value in extracted_data.items():
                    pricing_matrix = value.get("program_pricing_matrix", {})
                    total_base += parse_amount(
                        pricing_matrix.get("base_premium_usd", "")
                    )
                    total_terrorism += parse_amount(
                        pricing_matrix.get("terrorism_premium_usd", "")
                    )
                    total_all_in += parse_amount(
                        pricing_matrix.get("all_in_premium_usd", "")
                    )

                summary_of_pricing = {
                    "total_base_premium_usd": total_base,
                    "total_terrorism_premium_usd": total_terrorism,
                    "total_all_in_premium_usd": total_all_in,
                }

                data_mapping_dict[doc_type]["commercial_proposal_data_us"][
                    "summary_of_pricing"
                ] = summary_of_pricing
                proposal_comparison_data[unique_key]["summary_of_pricing"] = (
                    summary_of_pricing
                )

            logger.info(
                f"Document processing completed: {successful_docs} successful, {failed_docs} failed"
            )

            # If all documents failed, raise an error
            if failed_docs == len(to_extract) and to_extract:
                raise Exception("All document processing tasks failed")

            async def update_progress(state, progress):
                # Mark task as completed when reaching 100%
                if progress >= 100:
                    await redis_client.mark_task_completed(proposal_id)

                await redis_client.publish_task_update(
                    task_id=proposal_id,
                    state=state,
                    progress=progress,
                    additional_data={"proposal_id": proposal_id},
                )

            # with open("commercial_data_1.json", "w") as f:
            #     json.dump(data_mapping_dict, f, indent=4)

            # anomalies = await commercial_compare_jsons(
            #     policy=data_mapping_dict.get("Policy", {}).get("commercial_extracted_data") if data_mapping_dict.get("Policy").get("document_id") != "" else None,
            #     binder=data_mapping_dict.get("Binder", {}).get("commercial_extracted_data") if data_mapping_dict.get("Binder").get("document_id") != "" else None,
            #     quote=data_mapping_dict.get("Quote", {}).get("commercial_extracted_data") if data_mapping_dict.get("Quote").get("document_id") != "" else None
            # )

            carriers_list = []
            for carrier_key, carrier_data in data_mapping_dict.items():
                lob_details = carrier_data.get("commercial_proposal_data_us", {})
                insurer_name = None

                # Try to extract insurer name from first available LOB
                for lob_key, lob_val in lob_details.items():
                    try:
                        name = lob_val.get("program_pricing_matrix", {}).get(
                            "carrier", ""
                        )
                        if name and name.strip():
                            insurer_name = name.strip()
                            break
                    except AttributeError:
                        continue

                # Fallback to key if no insurer_name found
                carrier_name = insurer_name or carrier_key

                carriers_list.append(
                    {"carrier_name": carrier_name, "lob_details": lob_details}
                )

            formatted_data = {"carriers": carriers_list}

            try:
                logger.info("Generating carrier recommendation")
                recommendation = await recommend_best_carrier(formatted_data, context)
                proposal_comparison_data["recommendation"] = recommendation["result"]
                recommendation_tokens = recommendation.get("token", {})
                logger.info("Carrier recommendation completed")
            except Exception as e:
                logger.error(f"Carrier recommendation (US) FAILED: {str(e)}", exc_info=True)
                proposal_comparison_data["recommendation"] = {
                    "recommended_carrier": "N/A",
                    "recommendation_text": "Error generating recommendation. Please review comparison manually."
                }
                recommendation_tokens = {}

            logger.info("Storing proposal generation results")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://127.0.0.1:8000/api/v1/user/store/commercialproposal/data",
                    json={
                        "proposal_id": proposal_id,
                        "lob_type": list(unique_first_level_keys),
                        "data_mapping": data_mapping_dict,
                        "proposal_comparison_data": proposal_comparison_data,
                        "recommendation_tokens": recommendation_tokens,
                        "line_of_business": line_of_business,
                        "country": country,
                    },
                ) as resp:
                    store_response = await resp.json()
                    logger.info(f"Store API response status: {resp.status}")

            await update_progress("Proposal Generation Completed", 100)
            logger.info("Proposal generation (US) task completed successfully")

        except Exception as e:
            logger.error(f"Proposal generation (US) task FAILED: {str(e)}", exc_info=True)
            await redis_client.publish_task_failure(
                task_id=proposal_id,
                error_message=str(e),
                additional_data={"proposal_id": proposal_id},
            )
            raise

        finally:
            # Cleanup temporary files
            cleanup_tasks = []
            for value in data_mapping_dict.values():
                file_path = f"/tmp/{str(value.get('document_id'))}.pdf"
                if await asyncio.to_thread(os.path.exists, file_path):
                    cleanup_tasks.append(asyncio.to_thread(os.remove, file_path))
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)
                logger.info("Temporary files deleted successfully.")
            clear_task_id(proposal_id)

    # Run async task within the Celery worker's event loop
    async def run_task_with_timeout():
        try:
            return await asyncio.wait_for(run_task(), timeout=3600)  # 60 minutes timeout
        except asyncio.TimeoutError:
            logger.error(f"Task {proposal_id} timed out after 1 hour")
            await redis_client.publish_task_failure(
                task_id=proposal_id,
                error_message="Task completely stuck or timed out after 1 hour limit.",
                additional_data={"proposal_id": proposal_id},
            )
            raise Exception("Task timed out after 3600 seconds")

    return run_async_task_safely(run_task_with_timeout())