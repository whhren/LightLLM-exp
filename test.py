import asyncio
import time

async def async_hello_world():
    now = time.time()
    await asyncio.sleep(1)
    print(time.time() - now)
    print("Hello, world!")
    await asyncio.sleep(1)
    print(time.time() - now)

async def main():
    await asyncio.gather(async_hello_world(), async_hello_world(), async_hello_world())

now = time.time()
# run 3 async_hello_world() coroutine concurrently
asyncio.run(main())

print(f"Total time for running 3 coroutine: {time.time() - now}")

import time
def normal_hello_world():
    now = time.time()
    time.sleep(1)
    print(time.time() - now)
    print("Hello, world!")
    time.sleep(1)
    print(time.time() - now)

now = time.time()
normal_hello_world()
normal_hello_world()
normal_hello_world()
print(f"Total time for running 3 normal function: {time.time() - now}")