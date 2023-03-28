import math
from pygraphblas import Matrix
from pygraphblas.types import BOOL, INT64
from typing import List

from project.utils import is_undirected


def count_triangles_for_each_vertex(graph: Matrix) -> List[int]:
    if graph.type != BOOL:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.BOOL")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if not is_undirected(graph):
        raise ValueError("Unsupported graph type. Expected undirected graph")

    squared = graph.mxm(graph, semiring=INT64.PLUS_TIMES, mask=graph.S)
    triangles = squared.reduce_vector()
    return [math.ceil(triangles.get(i, default=0) / 2) for i in range(triangles.size)]


def count_triangles_cohen(graph: Matrix) -> int:
    if graph.type != BOOL:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.BOOL")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if not is_undirected(graph):
        raise ValueError("Unsupported graph type. Expected undirected graph")

    counts = graph.tril().mxm(graph.triu(), semiring=INT64.PLUS_TIMES, mask=graph)
    return math.ceil(counts.reduce_int() / 2)


def count_triangles_sandia(graph: Matrix) -> int:
    if graph.type != BOOL:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.BOOL")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    if not is_undirected(graph):
        raise ValueError("Unsupported graph type. Expected undirected graph")

    tril = graph.tril()
    counts = tril.mxm(tril, semiring=INT64.PLUS_TIMES, mask=tril)
    return counts.reduce_int()
