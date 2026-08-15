# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from pyrit.exceptions import EmptyResponseException, pyrit_target_retry
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

logger = logging.getLogger(__name__)

WebSocketMessage = str | bytes
ResponseParser = Callable[[WebSocketMessage], str | None]
MessageBuilder = Callable[[str], WebSocketMessage]


class WebsocketTarget(PromptTarget):
    """
    Send text prompts to a configurable WebSocket service.

    The target keeps one initialized WebSocket connection for each PyRIT
    conversation. Callers provide the service-specific initialization messages,
    prompt builder, and response parser.
    """

    RESPONSE_TIMEOUT_SECONDS: float = 30.0
    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(supports_multi_turn=True)
    )

    def __init__(
        self,
        *,
        endpoint: str,
        initialization_strings: list[WebSocketMessage],
        response_parser: ResponseParser,
        message_builder: MessageBuilder,
        discard_initial_messages: int = 1,
        response_timeout_seconds: float = RESPONSE_TIMEOUT_SECONDS,
        existing_convo: dict[str, ClientConnection] | None = None,
        max_requests_per_minute: int | None = None,
        custom_configuration: TargetConfiguration | None = None,
        **websockets_kwargs: Any,
    ) -> None:
        """
        Initialize the WebSocket target.

        Args:
            endpoint (str): WebSocket endpoint. Must use the ``ws://`` or ``wss://`` scheme.
            initialization_strings (list[str | bytes]): Messages to send when a connection opens.
            response_parser (ResponseParser): Function that returns response text or ``None`` for
                a frame that the target should ignore.
            message_builder (MessageBuilder): Function that converts prompt text to a WebSocket message.
            discard_initial_messages (int): Number of parsed messages to discard after initialization.
            response_timeout_seconds (float): Maximum time to wait for a parsed response.
            existing_convo (dict[str, ClientConnection] | None): Pre-initialized connections by
                PyRIT conversation ID.
            max_requests_per_minute (int | None): Maximum number of requests per minute.
            custom_configuration (TargetConfiguration | None): Override the default target configuration.
            websockets_kwargs (Any): Additional keyword arguments for ``websockets.connect``.

        Raises:
            ValueError: If endpoint or numeric arguments are invalid.
        """
        if not endpoint.startswith(("ws://", "wss://")):
            raise ValueError("endpoint must start with 'ws://' or 'wss://'.")
        if discard_initial_messages < 0:
            raise ValueError("discard_initial_messages must be nonnegative.")
        if response_timeout_seconds <= 0:
            raise ValueError("response_timeout_seconds must be positive.")

        super().__init__(
            endpoint=endpoint,
            max_requests_per_minute=max_requests_per_minute,
            custom_configuration=custom_configuration,
        )

        self._initialization_strings = initialization_strings
        self._response_parser = response_parser
        self._message_builder = message_builder
        self._discard_initial_messages = discard_initial_messages
        self._response_timeout_seconds = response_timeout_seconds
        self._existing_conversation = existing_convo if existing_convo is not None else {}
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._cleanup_complete = asyncio.Event()
        self._cleanup_complete.set()
        self._active_requests_complete = asyncio.Event()
        self._active_requests_complete.set()
        self._active_request_count = 0
        self._is_cleaning_up = False
        self._websockets_kwargs = websockets_kwargs

    async def connect_async(self) -> ClientConnection:
        """
        Open a connection to the configured WebSocket endpoint.

        Returns:
            ClientConnection: The open WebSocket connection.
        """
        logger.info("Connecting to WebSocket endpoint: %s", self._endpoint)
        connection = await websockets.connect(uri=self._endpoint, **self._websockets_kwargs)
        logger.info("Connected to WebSocket endpoint")
        return connection

    async def send_message_async(self, *, message: WebSocketMessage, conversation_id: str) -> None:
        """
        Send one message on an existing conversation connection.

        Args:
            message (str | bytes): Message to send.
            conversation_id (str): PyRIT conversation ID.
        """
        websocket = self._get_websocket(conversation_id=conversation_id)
        await websocket.send(message)

    async def receive_messages_async(self, conversation_id: str) -> str:
        """
        Receive frames until the response parser returns text.

        Args:
            conversation_id (str): PyRIT conversation ID.

        Returns:
            str: Parsed response text.
        """
        websocket = self._get_websocket(conversation_id=conversation_id)
        return await self._receive_message_async(websocket=websocket)

    async def send_text_async(self, *, text: str, conversation_id: str) -> str:
        """
        Send a text prompt and wait for its response.

        Args:
            text (str): Prompt text.
            conversation_id (str): PyRIT conversation ID.

        Returns:
            str: Parsed response text.

        Raises:
            TimeoutError: If no parsed response arrives before the configured timeout.
        """
        await self.send_message_async(
            message=self._message_builder(text),
            conversation_id=conversation_id,
        )
        try:
            return await asyncio.wait_for(
                self.receive_messages_async(conversation_id),
                timeout=self._response_timeout_seconds,
            )
        except TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for a WebSocket response after {self._response_timeout_seconds} seconds."
            ) from None

    async def cleanup_conversation_async(self, conversation_id: str) -> None:
        """
        Close and remove one conversation connection.

        Args:
            conversation_id (str): PyRIT conversation ID.
        """
        await self._begin_request_async()
        try:
            conversation_lock = self._conversation_locks.setdefault(conversation_id, asyncio.Lock())
            async with conversation_lock:
                websocket = self._existing_conversation.pop(conversation_id, None)
                if websocket is None:
                    return
                await websocket.close()
                logger.info("Disconnected WebSocket conversation: %s", conversation_id)
        finally:
            self._end_request()

    async def cleanup_target_async(self) -> None:
        """
        Close and remove all conversation connections.

        Raises:
            ConnectionError: If one or more connections cannot be closed.
        """
        await self._begin_cleanup_async()
        try:
            await self._active_requests_complete.wait()
            await self._close_all_connections_async()
            self._conversation_locks.clear()
        finally:
            await self._finish_cleanup_async()

    def is_json_response_supported(self) -> bool:
        """
        Check whether the target supports JSON response mode.

        Returns:
            bool: Always ``False`` because the service-specific parser controls the response format.
        """
        return False

    @limit_requests_per_minute
    @pyrit_target_retry
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Send the current normalized message to the WebSocket service.

        Args:
            normalized_conversation (list[Message]): Normalized conversation with the current request last.

        Returns:
            list[Message]: A list containing the target response.

        Raises:
            ValueError: If the current message is not text.
        """
        await self._begin_request_async()
        try:
            request = normalized_conversation[-1].message_pieces[0]
            if request.converted_value_data_type != "text":
                raise ValueError(f"Unsupported response type: {request.converted_value_data_type}")

            conversation_id = request.conversation_id
            conversation_lock = self._conversation_locks.setdefault(conversation_id, asyncio.Lock())

            async with conversation_lock:
                try:
                    await self._get_or_create_connection_async(conversation_id=conversation_id)
                    result = await self.send_text_async(
                        text=request.converted_value,
                        conversation_id=conversation_id,
                    )
                except BaseException:
                    await self._discard_connection_async(conversation_id=conversation_id)
                    raise
        finally:
            self._end_request()

        response_piece = construct_response_from_request(
            request=request,
            response_text_pieces=[result],
            response_type="text",
        ).message_pieces[0]
        return [Message(message_pieces=[response_piece])]

    async def _get_or_create_connection_async(self, *, conversation_id: str) -> ClientConnection:
        existing_connection = self._existing_conversation.get(conversation_id)
        if existing_connection is not None:
            return existing_connection

        websocket = await self.connect_async()
        try:
            await self._initialize_connection_async(websocket=websocket)
        except BaseException:
            try:
                await websocket.close()
            except Exception as error:
                logger.warning("Failed to close an uninitialized WebSocket connection: %s", error)
            raise

        self._existing_conversation[conversation_id] = websocket
        return websocket

    async def _initialize_connection_async(self, *, websocket: ClientConnection) -> None:
        for initialization_string in self._initialization_strings:
            await websocket.send(initialization_string)

        for _ in range(self._discard_initial_messages):
            try:
                await asyncio.wait_for(
                    self._receive_message_async(websocket=websocket),
                    timeout=self._response_timeout_seconds,
                )
            except TimeoutError:
                raise TimeoutError(
                    "Timed out waiting for an initial WebSocket message after "
                    f"{self._response_timeout_seconds} seconds."
                ) from None

    async def _receive_message_async(self, *, websocket: ClientConnection) -> str:
        async for message in websocket:
            parsed_message = self._response_parser(message)
            if parsed_message is None:
                continue
            if not parsed_message:
                raise EmptyResponseException(message="The WebSocket target returned an empty response.")
            return parsed_message

        raise ConnectionError("The WebSocket connection closed before a response was received.")

    async def _discard_connection_async(self, *, conversation_id: str) -> None:
        websocket = self._existing_conversation.pop(conversation_id, None)
        if websocket is None:
            return
        try:
            await websocket.close()
        except Exception as error:
            logger.warning("Failed to close unusable WebSocket conversation %s: %s", conversation_id, error)

    async def _begin_request_async(self) -> None:
        while True:
            async with self._lifecycle_lock:
                if not self._is_cleaning_up:
                    self._active_request_count += 1
                    self._active_requests_complete.clear()
                    return
                cleanup_complete = self._cleanup_complete
            await cleanup_complete.wait()

    def _end_request(self) -> None:
        self._active_request_count -= 1
        if self._active_request_count == 0:
            self._active_requests_complete.set()

    async def _begin_cleanup_async(self) -> None:
        while True:
            async with self._lifecycle_lock:
                if not self._is_cleaning_up:
                    self._is_cleaning_up = True
                    self._cleanup_complete.clear()
                    return
                cleanup_complete = self._cleanup_complete
            await cleanup_complete.wait()

    async def _finish_cleanup_async(self) -> None:
        async with self._lifecycle_lock:
            self._is_cleaning_up = False
            self._cleanup_complete.set()

    async def _close_all_connections_async(self) -> None:
        connections = list(self._existing_conversation.items())
        close_future = asyncio.gather(
            *(websocket.close() for _, websocket in connections),
            return_exceptions=True,
        )
        cancellation_error: asyncio.CancelledError | None = None

        try:
            close_results = await asyncio.shield(close_future)
        except asyncio.CancelledError as error:
            cancellation_error = error
            close_results = await close_future

        first_error: BaseException | None = None
        for (conversation_id, websocket), close_result in zip(connections, close_results, strict=True):
            if isinstance(close_result, BaseException):
                logger.error("Failed to close WebSocket conversation %s: %s", conversation_id, close_result)
                first_error = first_error or close_result
                continue
            if self._existing_conversation.get(conversation_id) is websocket:
                del self._existing_conversation[conversation_id]
            logger.info("Disconnected WebSocket conversation: %s", conversation_id)

        if cancellation_error is not None:
            raise cancellation_error
        if first_error is not None:
            raise ConnectionError("Failed to close one or more WebSocket connections.") from first_error

    def _get_websocket(self, *, conversation_id: str) -> ClientConnection:
        websocket = self._existing_conversation.get(conversation_id)
        if websocket is None:
            raise ConnectionError(f"WebSocket connection is not established for conversation {conversation_id}.")
        return websocket
