# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import pytest
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from pyrit.memory import SQLiteMemory
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import WebsocketTarget


@pytest.fixture
def response_parser() -> Callable[[str | bytes], str | None]:
    def parse_response(message: str | bytes) -> str | None:
        if isinstance(message, bytes):
            message = message.decode()
        return json.loads(message).get("message")

    return parse_response


@pytest.fixture
def message_builder() -> Callable[[str], str | bytes]:
    def build_message(prompt: str) -> str:
        return json.dumps({"message": prompt})

    return build_message


@pytest.fixture
def websocket_target(
    response_parser: Callable[[str | bytes], str | None],
    message_builder: Callable[[str], str | bytes],
    sqlite_instance: SQLiteMemory,
) -> WebsocketTarget:
    return WebsocketTarget(
        endpoint="wss://example.com",
        initialization_strings=["connect", "authenticate"],
        response_parser=response_parser,
        message_builder=message_builder,
        discard_initial_messages=0,
    )


def create_message(*, value: str = "Hello", conversation_id: str = "conversation") -> Message:
    return MessagePiece(
        original_value=value,
        original_value_data_type="text",
        converted_value=value,
        converted_value_data_type="text",
        role="user",
        conversation_id=conversation_id,
    ).to_message()


def test_init_invalid_endpoint_raises(
    response_parser: Callable[[str | bytes], str | None],
    message_builder: Callable[[str], str | bytes],
    sqlite_instance: SQLiteMemory,
) -> None:
    with pytest.raises(ValueError, match="endpoint must start"):
        WebsocketTarget(
            endpoint="https://example.com",
            initialization_strings=[],
            response_parser=response_parser,
            message_builder=message_builder,
        )


def test_init_invalid_discard_count_raises(
    response_parser: Callable[[str | bytes], str | None],
    message_builder: Callable[[str], str | bytes],
    sqlite_instance: SQLiteMemory,
) -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        WebsocketTarget(
            endpoint="wss://example.com",
            initialization_strings=[],
            response_parser=response_parser,
            message_builder=message_builder,
            discard_initial_messages=-1,
        )


def test_init_invalid_timeout_raises(
    response_parser: Callable[[str | bytes], str | None],
    message_builder: Callable[[str], str | bytes],
    sqlite_instance: SQLiteMemory,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        WebsocketTarget(
            endpoint="wss://example.com",
            initialization_strings=[],
            response_parser=response_parser,
            message_builder=message_builder,
            response_timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_connect_async_passes_websocket_arguments(
    response_parser: Callable[[str | bytes], str | None],
    message_builder: Callable[[str], str | bytes],
    sqlite_instance: SQLiteMemory,
) -> None:
    target = WebsocketTarget(
        endpoint="wss://example.com",
        initialization_strings=[],
        response_parser=response_parser,
        message_builder=message_builder,
        proxy="http://proxy.example.com",
    )
    connection = AsyncMock(spec=ClientConnection)

    with patch(
        "pyrit.prompt_target.websocket_target.websockets.connect",
        new_callable=AsyncMock,
        return_value=connection,
    ) as mock_connect:
        result = await target.connect_async()

    assert result is connection
    mock_connect.assert_awaited_once_with(uri="wss://example.com", proxy="http://proxy.example.com")


@pytest.mark.asyncio
async def test_send_prompt_async_initializes_connection_once(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)

    with (
        patch.object(websocket_target, "connect_async", new_callable=AsyncMock, return_value=connection) as connect,
        patch.object(
            websocket_target,
            "send_text_async",
            new_callable=AsyncMock,
            side_effect=["First response", "Second response"],
        ) as send_text,
    ):
        first_response = await websocket_target.send_prompt_async(
            message=create_message(value="First", conversation_id="shared")
        )
        second_response = await websocket_target.send_prompt_async(
            message=create_message(value="Second", conversation_id="shared")
        )

    connect.assert_awaited_once()
    assert connection.send.await_count == 2
    assert [call.args[0] for call in connection.send.await_args_list] == ["connect", "authenticate"]
    assert send_text.await_count == 2
    assert first_response[0].get_value() == "First response"
    assert second_response[0].get_value() == "Second response"

    await websocket_target.cleanup_target_async()


@pytest.mark.asyncio
async def test_send_prompt_async_serializes_same_conversation(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    active_requests = 0
    maximum_active_requests = 0

    async def send_text(*, text: str, conversation_id: str) -> str:
        nonlocal active_requests, maximum_active_requests
        active_requests += 1
        maximum_active_requests = max(maximum_active_requests, active_requests)
        await asyncio.sleep(0)
        active_requests -= 1
        return text

    with (
        patch.object(websocket_target, "connect_async", new_callable=AsyncMock, return_value=connection) as connect,
        patch.object(websocket_target, "send_text_async", side_effect=send_text),
    ):
        await asyncio.gather(
            websocket_target.send_prompt_async(message=create_message(value="First", conversation_id="shared")),
            websocket_target.send_prompt_async(message=create_message(value="Second", conversation_id="shared")),
        )

    connect.assert_awaited_once()
    assert maximum_active_requests == 1

    await websocket_target.cleanup_target_async()


@pytest.mark.asyncio
async def test_send_prompt_async_failure_discards_connection(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    websocket_target._existing_conversation["conversation"] = connection

    with patch.object(
        websocket_target,
        "send_text_async",
        new_callable=AsyncMock,
        side_effect=ConnectionError("connection failed"),
    ):
        with pytest.raises(ConnectionError, match="connection failed"):
            await websocket_target.send_prompt_async(message=create_message())

    connection.close.assert_awaited_once()
    assert "conversation" not in websocket_target._existing_conversation


def test_validate_request_invalid_type_raises(websocket_target: WebsocketTarget) -> None:
    message = MessagePiece(
        original_value="image.png",
        original_value_data_type="image_path",
        converted_value="image.png",
        converted_value_data_type="image_path",
        role="user",
    ).to_message()

    with pytest.raises(ValueError, match="supports only the following data types: text"):
        websocket_target._validate_request(normalized_conversation=[message])


@pytest.mark.asyncio
async def test_receive_messages_async_ignores_unparsed_frames(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    connection.__aiter__.return_value = [
        json.dumps({"event": "progress"}),
        json.dumps({"message": "response"}),
    ]
    websocket_target._existing_conversation["conversation"] = connection

    result = await websocket_target.receive_messages_async("conversation")

    assert result == "response"


@pytest.mark.asyncio
async def test_receive_messages_async_propagates_parser_error(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    connection.__aiter__.return_value = ["not-json"]
    websocket_target._existing_conversation["conversation"] = connection

    with pytest.raises(json.JSONDecodeError):
        await websocket_target.receive_messages_async("conversation")


@pytest.mark.asyncio
async def test_receive_messages_async_propagates_connection_closed(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    close_frame = Close(1000, "Normal closure")

    class FailingAsyncIterator:
        def __aiter__(self) -> "FailingAsyncIterator":
            return self

        async def __anext__(self) -> str:
            raise ConnectionClosed(rcvd=close_frame, sent=None)

    connection.__aiter__.side_effect = lambda: FailingAsyncIterator()
    websocket_target._existing_conversation["conversation"] = connection

    with pytest.raises(ConnectionClosed):
        await websocket_target.receive_messages_async("conversation")


@pytest.mark.asyncio
async def test_send_text_async_timeout_raises(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    websocket_target._existing_conversation["conversation"] = connection
    websocket_target._response_timeout_seconds = 0.001

    async def wait_forever(conversation_id: str) -> str:
        await asyncio.sleep(1)
        return "unreachable"

    with patch.object(websocket_target, "receive_messages_async", side_effect=wait_forever):
        with pytest.raises(TimeoutError, match="Timed out waiting for a WebSocket response"):
            await websocket_target.send_text_async(text="Hello", conversation_id="conversation")


@pytest.mark.asyncio
async def test_cleanup_target_async_attempts_every_connection(websocket_target: WebsocketTarget) -> None:
    failing_connection = AsyncMock(spec=ClientConnection)
    failing_connection.close.side_effect = RuntimeError("close failed")
    successful_connection = AsyncMock(spec=ClientConnection)
    websocket_target._existing_conversation = {
        "failing": failing_connection,
        "successful": successful_connection,
    }

    with pytest.raises(ConnectionError, match="one or more"):
        await websocket_target.cleanup_target_async()

    failing_connection.close.assert_awaited_once()
    successful_connection.close.assert_awaited_once()
    assert websocket_target._existing_conversation == {"failing": failing_connection}


@pytest.mark.asyncio
async def test_cleanup_target_async_waits_for_active_send(websocket_target: WebsocketTarget) -> None:
    connection = AsyncMock(spec=ClientConnection)
    websocket_target._existing_conversation["conversation"] = connection
    send_started = asyncio.Event()
    finish_send = asyncio.Event()

    async def send_text(*, text: str, conversation_id: str) -> str:
        send_started.set()
        await finish_send.wait()
        return text

    with patch.object(websocket_target, "send_text_async", side_effect=send_text):
        send_task = asyncio.create_task(websocket_target.send_prompt_async(message=create_message()))
        await send_started.wait()
        cleanup_task = asyncio.create_task(websocket_target.cleanup_target_async())
        await asyncio.sleep(0)

        connection.close.assert_not_awaited()
        assert not cleanup_task.done()

        finish_send.set()
        await send_task
        await cleanup_task

    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_target_async_cancellation_finishes_closing_connections(
    websocket_target: WebsocketTarget,
) -> None:
    connection = AsyncMock(spec=ClientConnection)
    websocket_target._existing_conversation["conversation"] = connection
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    async def close_connection() -> None:
        close_started.set()
        await finish_close.wait()

    connection.close.side_effect = close_connection
    cleanup_task = asyncio.create_task(websocket_target.cleanup_target_async())
    await close_started.wait()

    cleanup_task.cancel()
    await asyncio.sleep(0)
    assert not cleanup_task.done()

    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert websocket_target._existing_conversation == {}
