# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid

import pytest
from websockets.asyncio.server import ServerConnection, serve

from pyrit.memory import SQLiteMemory
from pyrit.models import MessagePiece
from pyrit.prompt_target import WebsocketTarget


@pytest.mark.asyncio
@pytest.mark.run_only_if_all_tests
async def test_websocket_target_round_trip_with_local_pyrit_server(sqlite_instance: SQLiteMemory) -> None:
    received_messages: list[dict[str, str]] = []

    async def pyrit_websocket_handler(websocket: ServerConnection) -> None:
        initialization = json.loads(await websocket.recv())
        received_messages.append(initialization)
        await websocket.send(json.dumps({"message": "PyRIT WebSocket target ready"}))

        async for raw_message in websocket:
            prompt_message = json.loads(raw_message)
            received_messages.append(prompt_message)
            await websocket.send(json.dumps({"event": "processing"}))
            await websocket.send(json.dumps({"message": f"PyRIT received: {prompt_message['prompt']}"}))

    def response_parser(message: str | bytes) -> str | None:
        if isinstance(message, bytes):
            message = message.decode()
        return json.loads(message).get("message")

    def message_builder(prompt: str) -> str:
        return json.dumps({"type": "prompt", "prompt": prompt})

    async with serve(pyrit_websocket_handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        target = WebsocketTarget(
            endpoint=f"ws://127.0.0.1:{port}",
            initialization_strings=[json.dumps({"type": "initialize", "client": "PyRIT"})],
            response_parser=response_parser,
            message_builder=message_builder,
            discard_initial_messages=1,
        )

        conversation_id = str(uuid.uuid4())
        request = MessagePiece(
            role="user",
            original_value="Hello",
            original_value_data_type="text",
            conversation_id=conversation_id,
        ).to_message()

        try:
            response = await target.send_prompt_async(message=request)
        finally:
            await target.cleanup_target_async()

    assert response[0].get_value() == "PyRIT received: Hello"
    assert received_messages == [
        {"type": "initialize", "client": "PyRIT"},
        {"type": "prompt", "prompt": "Hello"},
    ]
