# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
# ---

# %% [markdown]
# # WebSocket Target
#
# `WebsocketTarget` connects PyRIT to services that use a custom WebSocket protocol.
# Supply the service-specific initialization messages, prompt builder, and response parser.
#
# This example starts a local PyRIT WebSocket service. It exercises the real WebSocket
# transport without credentials or an external endpoint.

# %%
import json

from websockets.asyncio.server import ServerConnection, serve

from pyrit.models import MessagePiece
from pyrit.prompt_target import WebsocketTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore


async def pyrit_websocket_handler(websocket: ServerConnection) -> None:
    initialization = json.loads(await websocket.recv())
    if initialization != {"type": "initialize", "client": "PyRIT"}:
        await websocket.close(code=1002, reason="Invalid initialization message")
        return

    await websocket.send(json.dumps({"message": "PyRIT WebSocket target ready"}))

    async for raw_message in websocket:
        request = json.loads(raw_message)
        await websocket.send(json.dumps({"event": "processing"}))
        await websocket.send(json.dumps({"message": f"PyRIT received: {request['prompt']}"}))


def response_parser(message: str | bytes) -> str | None:
    if isinstance(message, bytes):
        message = message.decode()
    return json.loads(message).get("message")


def message_builder(prompt: str) -> str:
    return json.dumps({"type": "prompt", "prompt": prompt})


# %% [markdown]
# Start the local service on an available loopback port, then configure the target for its protocol.

# %%
server = await serve(pyrit_websocket_handler, "127.0.0.1", 0)  # type: ignore
port = server.sockets[0].getsockname()[1]

target = WebsocketTarget(
    endpoint=f"ws://127.0.0.1:{port}",
    initialization_strings=[json.dumps({"type": "initialize", "client": "PyRIT"})],
    response_parser=response_parser,
    message_builder=message_builder,
    discard_initial_messages=1,
)

# %% [markdown]
# Send a prompt through the target and close both sides of the connection.

# %%
request = MessagePiece(
    role="user",
    original_value="Hello",
    original_value_data_type="text",
).to_message()

try:
    response = await target.send_prompt_async(message=request)  # type: ignore
    print(response[0].get_value())
finally:
    await target.cleanup_target_async()  # type: ignore
    server.close()
    await server.wait_closed()  # type: ignore
