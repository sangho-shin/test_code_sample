import asyncio
import random

async def async_hello():
    print("hello, world!")

async def print_number(number):
    await asyncio.sleep(random.random())
    print(number)


# loop = asyncio.get_event_loop()
# loop.run_until_complete(async_hello())
#
# loop.close()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    loop.run_until_complete(
        asyncio.gather(
            *[
                print_number(number) for number in range(10)
            ]
        )
    )
    loop.close()