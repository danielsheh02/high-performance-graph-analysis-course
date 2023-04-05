import math
from typing import List, Tuple

from pygraphblas import Matrix, FP64


def sssp(graph: Matrix, start_vertex: int) -> List[int]:
    """
    The function searches for the shortest path in a directed graph from a given start vertex.

    :param graph: the graph is represented as an adjacency matrix
    :param start_vertex: the starting vertex of the graph
    :return: returns a list of values where the distance to it
    from the starting vertex is specified for each vertex
    """
    return mssp(graph, [start_vertex])[0][1]


def mssp(graph: Matrix, start_vertices: List[int]) -> List[Tuple[int, List[int]]]:
    """
    The function searches for shortest paths in a directed graph from several given starting vertices.

    :param graph: the graph is represented as an adjacency matrix
    :param start_vertices: the starting vertices of the graph
    :return: returns a list of pairs: a vertex, and an array, where for each vertex
    the distance to it from the specified one is specified. If the vertex is
    not reachable, then the value of the corresponding cell is float('inf').
    """
    if graph.type != FP64:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.FP64")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    for i in range(graph.nrows):
        graph[i, i] = 0
    dists = Matrix.sparse(FP64, nrows=len(start_vertices), ncols=graph.ncols)
    for row, start in enumerate(start_vertices):
        if start < 0 or start >= graph.nrows:
            raise ValueError("No vertex with such number")
        dists[row, start] = 0

    for _ in range(graph.nrows - 1):
        dists.mxm(graph, semiring=FP64.MIN_PLUS, out=dists)

    if dists.isne(dists.mxm(graph, semiring=FP64.MIN_PLUS)):
        raise ValueError("Graph has a negative weight cycle")

    return [
        (start, [dists.get(i, j, default=math.inf) for j in range(graph.nrows)])
        for i, start in enumerate(start_vertices)
    ]


def floyd_warshall(graph: Matrix) -> List[Tuple[int, List[int]]]:
    """
    The function searches for shortest paths in a directed graph for all pairs of vertices.

    :param graph: the graph is represented as an adjacency matrix
    :return: returns a list of pairs: a vertex, and an array, where for each vertex
    the distance to it from the specified one is specified. If the vertex is
    not reachable, then the value of the corresponding cell is float('inf').
    """
    if graph.type != FP64:
        raise ValueError("Unsupported graph type. Expected type pygraphblas.FP64")
    if not graph.square:
        raise ValueError("Adjacency matrix of the graph must be square")
    for i in range(graph.nrows):
        graph[i, i] = 0
    dists = graph.dup()

    for k in range(graph.nrows):
        step = dists.extract_matrix(col_index=k).mxm(
            dists.extract_matrix(row_index=k), semiring=FP64.MIN_PLUS
        )
        dists.eadd(step, add_op=FP64.MIN, out=dists)

    for k in range(graph.nrows):
        step = dists.extract_matrix(col_index=k).mxm(
            dists.extract_matrix(row_index=k), semiring=FP64.MIN_PLUS
        )
        if dists.isne(dists.eadd(step, add_op=FP64.MIN)):
            raise ValueError("Graph has a negative weight cycle")

    return [
        (row, [dists.get(row, col, default=math.inf) for col in range(graph.nrows)])
        for row in range(graph.nrows)
    ]
