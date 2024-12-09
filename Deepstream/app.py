import sys
sys.path.append("../")
import asyncio
from Shared.services.queue.consumer import Consumer

import time
from dependency_injector.wiring import Provide, inject
from .resources import Container,init_resources

async def main():
    container=await init_resources();
    container.wire(modules=[__name__])
    await listen()

@inject
async def listen(consumer:Consumer=Provide[Container.consumer]):
    consumer.start_consuming()
    while True:
        print("Listening...")
        await asyncio.sleep(1000)  # Non-blocking sleep
if __name__ == "__main__":
    asyncio.run(main())