from pygraphblas import Matrix, Vector
from pygraphblas.descriptor import RSC
from pygraphblas.types import BOOL, INT64
from typing import List, Tuple


def bfs(graph: Matrix, start_vertex: int) -> List[int]:
    """
    The function traverses the graph in breadth and
    calculates at which step which vertex is reachable.

    :param graph: the graph is represented as an adjacency matrix.
    :param start_vertex: the initial vertex of the graph
    :return: returns a list with the numbers of steps at which the vertices are reachable,
    the initial vertex is reachable at step 0, if the vertex is not reachable, -1 is returned
    """
    if graph.type != BOOL:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.BOOL")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if start_vertex < 0 or start_vertex >= graph.nrows:
        raise ValueError("No vertex with such number")

    steps = Vector.sparse(INT64, size=graph.nrows)
    front = Vector.sparse(BOOL, size=graph.nrows)
    steps[start_vertex] = 0
    front[start_vertex] = True

    step = 1
    while front.nvals > 0:
        front.vxm(graph, out=front, mask=steps.S, desc=RSC)
        steps.assign_scalar(step, mask=front)
        step += 1
    return [steps.get(i, -1) for i in range(steps.size)]


def msbfs(graph: Matrix, starts: List[int]) -> List[Tuple[int, List[int]]]:
    """
    The function traverses the graph in breadth and calculates from which vertex
    which is reachable by the shortest path.

    :param graph: the graph is represented as an adjacency matrix.
    :param starts: list of the initial vertices of the graph
    :return: returns a list of pairs where the 1st element is the starting vertex,
    the 2nd element is a list where for each vertex of the graph it is indicated
    from which vertex we came to this one by the shortest path from the starting vertex.
    """
    if graph.type != BOOL:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.BOOL")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")

    input_source = Matrix.sparse(INT64, nrows=len(starts), ncols=graph.ncols)
    front = Matrix.sparse(INT64, nrows=len(starts), ncols=graph.ncols)
    for row, start in enumerate(starts):
        if start < 0 or start >= graph.ncols:
            raise ValueError("No vertex with such number")
        input_source[row, start] = -1
        front[row, start] = start

    while front.nvals > 0:
        front.mxm(
            graph, out=front, semiring=INT64.MIN_FIRST, mask=input_source.S, desc=RSC
        )
        input_source.assign(front, mask=front.S)
        front.apply(INT64.POSITIONJ, out=front)

    return [
        (start, [input_source.get(row, col, default=-2) for col in range(graph.ncols)])
        for row, start in enumerate(starts)
    ]
