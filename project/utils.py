from pygraphblas import Matrix


def is_undirected(graph: Matrix) -> bool:
    print()
    for i, j in zip(graph.I, graph.J):
        if graph.get(i, j) != graph.get(j, i):
            return False
    return True
