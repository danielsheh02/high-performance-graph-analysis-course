import pytest
from project import *
from tests.utils import read_graphs_from_json


@pytest.mark.parametrize(
    "I, J, size, start_vertex, expected",
    read_graphs_from_json(
        "test_bfs",
        lambda data: (
            data["I"],
            data["J"],
            data["size"],
            data["start"],
            data["expected"],
        ),
    ),
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
