from concurrent.futures import ThreadPoolExecutor

def loudly_return():
    print("processing")
    return 42

with ThreadPoolExecutor(1) as executor:
    future = executor.submit(loudly_return)

print(future)

print(future.result())