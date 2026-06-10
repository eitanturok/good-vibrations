import asyncio
import time

from async_graph_data_flow import AsyncGraph, AsyncExecutor


async def func1():
    for i in range(2):
        await asyncio.sleep(1)
        print(f"At func1: {i}")
        yield f"From func1: {i}"


async def func2(data):
    print(f"At func2: {data}")
    yield f"From func2: {data}"


async def func3(data):
    print(f"At func3: {data}")
    yield data, "2nd arg from func3"


async def func4(data):
    print(f"At func4: {data}")
    yield data, "2nd arg from func4"


async def func5(data1, data2):
    print(f"At func5: {data1} + {data2}")
    yield


if __name__ == "__main__":
    graph = AsyncGraph()

    graph.add_node(func1)
    graph.add_node(func2)
    graph.add_node(func3)
    graph.add_node(func4)
    graph.add_node(func5)
    graph.add_edge("func1", "func2")
    graph.add_edge("func2", "func3")
    graph.add_edge("func2", "func4")
    graph.add_edge("func3", "func5")
    graph.add_edge("func4", "func5")

    print(f"Graph: {graph.nodes_to_edges}")

    executor = AsyncExecutor(graph)

    t1 = time.time()
    executor.execute()
    t2 = time.time()
    print(f"execution time:", t2 - t1)
