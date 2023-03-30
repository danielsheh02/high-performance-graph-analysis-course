import pytest
from project.triangles import *
from tests.utils import read_graphs_from_json


@pytest.mark.parametrize(
    "I, J, size, expected",
    read_graphs_from_json(
        "test_triangles_for_each_vertex",
        lambda data: (
            data["I"],
            data["J"],
            data["size"],
            data["expected"],
        ),
        "graphs_triangles.json",
    ),
)
def test_count_triangles_for_each_vertex(I, J, size, expected):
    graph = Matrix.from_lists(I, J, nrows=size, ncols=size)
    assert count_triangles_for_each_vertex(graph) == expected


@pytest.mark.parametrize(
    "I, J, size, expected",
    read_graphs_from_json(
        "test_count_triangles",
        lambda data: (
            data["I"],
            data["J"],
            data["size"],
            data["expected"],
        ),
        "graphs_triangles.json",
    ),
)
def test_count_triangles(I, J, size, expected):
    graph = Matrix.from_lists(I, J, nrows=size, ncols=size)
    assert count_triangles_cohen(graph) == expected
    assert count_triangles_sandia(graph) == expected


def test_non_square_adj_matrix():
    graph = Matrix.dense(BOOL, 2, 3)
    with pytest.raises(ValueError):
        count_triangles_for_each_vertex(graph)
        count_triangles_cohen(graph)
        count_triangles_sandia(graph)


def test_unsupported_type_matrix():
    graph = Matrix.dense(INT64, 3, 3)
    with pytest.raises(ValueError):
        count_triangles_for_each_vertex(graph)
        count_triangles_cohen(graph)
        count_triangles_sandia(graph)
    graph_directed = Matrix.from_lists([0, 0, 2], [0, 1, 1], nrows=3, ncols=3)
    with pytest.raises(ValueError):
        count_triangles_for_each_vertex(graph_directed)
        count_triangles_cohen(graph_directed)
        count_triangles_sandia(graph_directed)
