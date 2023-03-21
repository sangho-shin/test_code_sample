from multiprocessing import Process, Pipe
import os


def work(identifier):
    print(f"I am the process, {identifier}, pid: {os.getpid()}")


def main():
    processes = [
        Process(target=work, args=(number,)) for number in range(5)
    ]

    for process in processes:
        process.start()

    while processes:
        processes.pop().join()

def worker(connection):
    while True:
        instance = connection.recv()

        if instance:
            print(f"CHLD: recv : {instance}")
        if instance is None:
            break

class CustomClass:
    pass

def main2():
    parent_conn, child_conn = Pipe()

    child = Process(target=worker, args=(child_conn,))

    for item in (
        42,
        "some string",
        {"one" : 1},
        CustomClass(),
        None
    ):

        print("PRNT : send : {}".format(item))
        parent_conn.send(item)

    child.start()
    child.join()


if __name__ == "__main__":
    # main()
    main2()

# work(1)
# main()