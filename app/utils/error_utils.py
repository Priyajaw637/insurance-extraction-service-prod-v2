import asyncio
from typing import Any, Dict, Optional

from fastapi import HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.modules.async_redis import redis_client
from app.utils.logger_factory import get_logger

logger = get_logger(__name__)


class ErrorHandler:
    """Centralized error handling and logging."""

    @staticmethod
    def is_websocket_connected(websocket: WebSocket) -> bool:
        """
        Check if WebSocket connection is still active.

        Args:
            websocket: WebSocket connection to check

        Returns:
            True if connection is active, False otherwise
        """
        try:
            return websocket.client_state == WebSocketState.CONNECTED
        except Exception:
            return False

    @staticmethod
    async def safe_websocket_send(
        websocket: WebSocket,
        data: Dict[str, Any],
        lead_id: str,
        context: str = "WebSocket operation",
    ) -> bool:
        """
        Safely send data to WebSocket with connection state validation.

        Args:
            websocket: WebSocket connection
            data: Data to send
            lead_id: Lead identifier for logging
            context: Context description for logging

        Returns:
            True if sent successfully, False otherwise
        """
        if not ErrorHandler.is_websocket_connected(websocket):
            logger.debug(
                f"Skipping WebSocket send for lead_id {lead_id} - connection not active"
            )
            return False

        try:
            await websocket.send_json(data)
            return True
        except Exception as send_error:
            logger.error(
                f"Failed to send {context} message to WebSocket for lead_id {lead_id}: {send_error}"
            )
            return False

    @staticmethod
    async def handle_websocket_error(
        websocket: WebSocket,
        error: Exception,
        lead_id: str,
        context: str = "WebSocket operation",
    ) -> bool:
        """
        Handle WebSocket errors with proper logging and client notification.

        Args:
            websocket: WebSocket connection
            error: The exception that occurred
            lead_id: Lead identifier for logging
            context: Context description for the error

        Returns:
            True if this is a disconnect error (should break processing loop), False otherwise
        """
        error_message = str(error)

        if isinstance(error, WebSocketDisconnect):
            logger.info(
                f"WebSocket disconnected for lead_id {lead_id} with code {error.code}: {error.reason or 'No reason provided'}"
            )
            return True  # Signal that processing should stop

        logger.error(
            f"Error in {context} for lead_id {lead_id}: {error_message}", exc_info=True
        )

        # Use safe sending mechanism to avoid sending to closed connections
        await ErrorHandler.safe_websocket_send(
            websocket, {"error": f"{context} failed: {error_message}"}, lead_id, context
        )

        return False  # Continue processing for non-disconnect errors

    @staticmethod
    async def safe_websocket_receive(
        websocket: WebSocket, lead_id: str, timeout_seconds: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Safely receive JSON data from WebSocket with connection state validation.

        Args:
            websocket: WebSocket connection
            lead_id: Lead identifier for logging
            timeout_seconds: Timeout for receive operation

        Returns:
            Received data if successful, None if connection lost or error
        """
        if not ErrorHandler.is_websocket_connected(websocket):
            logger.debug(
                f"Skipping WebSocket receive for lead_id {lead_id} - connection not active"
            )
            return None

        try:
            # Use asyncio.wait_for to add timeout protection
            import asyncio

            message = await asyncio.wait_for(
                websocket.receive_json(), timeout=timeout_seconds
            )
            return message
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during receive for lead_id {lead_id}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket receive timeout for lead_id {lead_id}")
            return None
        except Exception as e:
            logger.error(
                f"Error receiving WebSocket message for lead_id {lead_id}: {e}"
            )
            return None

    @staticmethod
    async def handle_websocket_close(
        websocket: WebSocket, lead_id: str, reason: str = "Internal server error"
    ) -> None:
        """
        Safely close WebSocket connection with error handling.

        Args:
            websocket: WebSocket connection to close
            lead_id: Lead identifier for logging
            reason: Reason for closing the connection
        """
        if not ErrorHandler.is_websocket_connected(websocket):
            logger.debug(
                f"WebSocket for lead_id {lead_id} already closed or disconnected"
            )
            return

        try:
            await websocket.close(code=1011, reason=reason)
        except Exception as close_error:
            logger.error(
                f"Failed to close WebSocket for lead_id {lead_id}: {close_error}"
            )

    @staticmethod
    async def handle_task_error(
        task_id: str, error: Exception, context: str = "Task execution"
    ) -> None:
        """
        Handle task errors with Redis notification.

        Args:
            task_id: Task identifier
            error: The exception that occurred
            context: Context description for the error
        """
        error_message = str(error)
        logger.error(
            f"Error in {context} for task {task_id}: {error_message}", exc_info=True
        )

        try:
            await redis_client.publish_task_update(
                task_id=task_id, state="Failed", progress=100, error=error_message
            )
        except Exception as redis_error:
            logger.error(
                f"Failed to publish error update to Redis for task {task_id}: {redis_error}"
            )

    @staticmethod
    def create_http_error_response(
        status_code: int, message: str, details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """
        Create standardized HTTP error response.

        Args:
            status_code: HTTP status code
            message: Error message
            details: Additional error details

        Returns:
            HTTPException with standardized format
        """
        error_detail = {"message": message}
        if details:
            error_detail.update(details)

        return HTTPException(status_code=status_code, detail=error_detail)

    @staticmethod
    def log_function_entry(func_name: str, **kwargs) -> None:
        """
        Log function entry with parameters.

        Args:
            func_name: Name of the function
            **kwargs: Function parameters to log
        """
        # Only log function entry for critical business operations
        if any(
            key in func_name.lower()
            for key in ["process", "extract", "compare", "generate"]
        ):
            logger.info(f"Starting {func_name}")
        else:
            logger.debug(f"Entering {func_name} with parameters: {kwargs}")

    @staticmethod
    async def safe_async_operation(
        operation,
        error_message: str,
        default_return: Any = None,
        log_errors: bool = True,
    ) -> Any:
        """
        Safely execute an async operation with error handling.

        Args:
            operation: Async operation to execute
            error_message: Error message prefix for logging
            default_return: Default value to return on error
            log_errors: Whether to log errors

        Returns:
            Operation result or default_return on error
        """
        try:
            return await operation
        except Exception as e:
            if log_errors:
                logger.error(f"{error_message}: {str(e)}", exc_info=True)
            return default_return

    @staticmethod
    async def safe_async_operation(
        operation,
        error_message: str,
        default_return: Any = None,
        log_errors: bool = True,
    ) -> Any:
        """
        Safely execute an async operation with error handling.

        Args:
            operation: Async operation to execute (coroutine or callable returning coroutine)
            error_message: Error message prefix for logging
            default_return: Default value to return on error
            log_errors: Whether to log errors

        Returns:
            Operation result or default_return on error
        """
        try:
            if asyncio.iscoroutine(operation):
                return await operation
            else:
                return await operation()
        except Exception as e:
            if log_errors:
                logger.error(f"{error_message}: {str(e)}", exc_info=True)
            return default_return

    @staticmethod
    def safe_sync_operation(
        operation,
        error_message: str,
        default_return: Any = None,
        log_errors: bool = True,
    ) -> Any:
        """
        Safely execute a sync operation with error handling.
        DEPRECATED: Use safe_async_operation for new code.

        Args:
            operation: Sync operation to execute
            error_message: Error message prefix for logging
            default_return: Default value to return on error
            log_errors: Whether to log errors

        Returns:
            Operation result or default_return on error
        """
        try:
            return operation()
        except Exception as e:
            if log_errors:
                logger.error(f"{error_message}: {str(e)}", exc_info=True)
            return default_return

    @staticmethod
    def format_validation_error(field_name: str, error_details: str) -> Dict[str, Any]:
        """
        Format validation error for consistent response structure.

        Args:
            field_name: Name of the field that failed validation
            error_details: Details about the validation error

        Returns:
            Formatted validation error dictionary
        """
        return {
            "type": "validation_error",
            "field": field_name,
            "message": error_details,
            "timestamp": None,  # Simplified timestamp handling
        }

    @staticmethod
    def log_business_event(
        event_type: str, details: Dict[str, Any] = None, user_id: str = None
    ) -> None:
        """
        Log important business events for audit and monitoring.

        Args:
            event_type: Type of business event (e.g., 'document_processed', 'form_completed')
            details: Additional event details
            user_id: User identifier for the event
        """
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            "details": details or {},
        }
        logger.info(f"Business Event: {event_type}", extra=event_data)

    @staticmethod
    def log_performance_metric(
        operation: str, duration: float, details: Dict[str, Any] = None
    ) -> None:
        """
        Log performance metrics for monitoring.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            details: Additional performance details
        """
        perf_data = {
            "operation": operation,
            "duration_seconds": duration,
            "details": details or {},
        }
        logger.info(
            f"Performance: {operation} completed in {duration:.3f}s", extra=perf_data
        )

    @staticmethod
    def log_security_event(
        event_type: str, details: Dict[str, Any] = None, severity: str = "warning"
    ) -> None:
        """
        Log security-related events.

        Args:
            event_type: Type of security event
            details: Event details
            severity: Severity level (info, warning, error, critical)
        """
        security_data = {
            "security_event": event_type,
            "severity": severity,
            "details": details or {},
        }

        if severity == "critical":
            logger.critical(f"Security Event: {event_type}", extra=security_data)
        elif severity == "error":
            logger.error(f"Security Event: {event_type}", extra=security_data)
        elif severity == "warning":
            logger.warning(f"Security Event: {event_type}", extra=security_data)
        else:
            logger.info(f"Security Event: {event_type}", extra=security_data)


# Global error handler instance
error_handler = ErrorHandler()
