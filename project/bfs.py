from pygraphblas import Matrix, Vector
from pygraphblas.descriptor import RSC
from pygraphblas.types import BOOL, INT64
from typing import List


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
    while front.nvals != 0:
        front.vxm(graph, out=front, mask=steps.S, desc=RSC)
        steps.assign_scalar(step, mask=front)
        step += 1
    return [steps.get(i, -1) for i in range(steps.size)]
