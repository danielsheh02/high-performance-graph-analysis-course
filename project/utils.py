from pygraphblas import Matrix


def is_undirected(graph: Matrix) -> bool:
    """
    The function determines from the adjacency matrix
    whether the graph is undirected.

    :param graph: the graph is represented as an adjacency matrix.
    :return: returns a boolean value, is it true that
    the graph is undirected.
    """
    for i, j in zip(graph.I, graph.J):
        if graph.get(i, j) != graph.get(j, i):
            return False
    return True
