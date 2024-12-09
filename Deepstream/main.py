#!/usr/bin/env python3
import sys
import gi
gi.require_version('Gst', '1.0')
from ctypes import *
import sys
import asyncio 
from dependency_injector.wiring import Provide, inject
from resources import Container,init_resources
from deepstream_consumer import DeepstreamConsumer
from ctypes import CDLL

# Initialize X11 for thread safety
x11 = CDLL("libX11.so")
x11.XInitThreads()
async def main():
    container=await init_resources();
    container.wire(modules=[__name__])

    await listen()



@inject
async def listen(consumer:DeepstreamConsumer=Provide[Container.consumer]):
    consumer.start_consuming()
    while True:
        print("Listening...")
        await asyncio.sleep(1000)  # Non-blocking sleep

if __name__ == "__main__":
    asyncio.run(main())