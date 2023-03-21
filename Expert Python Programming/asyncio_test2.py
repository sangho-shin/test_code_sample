import time
import random
import asyncio

async def waiter(name):
    for _ in range(4):
        time_to_sleep = random.randint(1, 3) / 4
        # 함수의 실행을 블록 함
        # time.sleep(time_to_sleep)
        # 비 동기 버전
        await asyncio.sleep(time_to_sleep)
        print(f"{name} waited {time_to_sleep} seconds")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(waiter("first"), waiter("second"))
    )
