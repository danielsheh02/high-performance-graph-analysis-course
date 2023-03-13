from pygraphblas import Matrix, Vector
from pygraphblas.types import BOOL, INT64
from typing import List


def bfs(graph: Matrix, start_vertex: int) -> List[int]:
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
    old_steps_nvals = -1
    while old_steps_nvals != steps.nvals:
        old_steps_nvals = steps.nvals
        front.vxm(graph, out=front)
        current_visited_mask = front.eadd(steps.S, add_op=BOOL.GT, mask=front.S)
        steps.assign_scalar(step, mask=current_visited_mask)
        step += 1
    return [steps.get(i, -1) for i in range(steps.size)]
