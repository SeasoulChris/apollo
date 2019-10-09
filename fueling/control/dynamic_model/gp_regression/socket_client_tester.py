#!/usr/bin/env python

# WS client example

import asyncio
import websockets
from datetime import datetime
import time

async def hello():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        await websocket.send(name)
        print(f"> {name}")

        greeting = await websocket.recv()
        print(f"< {greeting}")
while True:
    asyncio.get_event_loop().run_until_complete(hello())
    time.sleep(0.01)
