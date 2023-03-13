import pytest
from project import *


@pytest.mark.parametrize(
    "I, J, size, start_vertex, expected",
    [
        (
            [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6],
            [1, 3, 6, 4, 5, 0, 2, 5, 2, 3, 4, 2],
            7,
            6,
            [2, 3, 1, 1, 1, 2, 0],
        ),
        (
            [0, 1, 2, 2, 4],
            [1, 2, 0, 3, 2],
            5,
            0,
            [0, 1, 2, 3, -1],
        ),
        (
            [0, 0, 2],
            [0, 1, 1],
            3,
            0,
            [0, 1, -1],
        ),
    ],
)
def test_bfs(I, J, size, start_vertex, expected):
    graph = Matrix.from_lists(I, J, nrows=size, ncols=size)
    assert expected == bfs(graph, start_vertex)


def test_non_square_adj_matrix():
    graph = Matrix.dense(BOOL, 2, 3)
    with pytest.raises(ValueError):
        bfs(graph, 0)


def test_unsupported_type_matrix():
    graph = Matrix.dense(INT64, 3, 3)
    with pytest.raises(ValueError):
        bfs(graph, 0)


def test_incorrect_start_node():
    graph = Matrix.dense(BOOL, 3, 3)
    with pytest.raises(ValueError):
        bfs(graph, 3)
