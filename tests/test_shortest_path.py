import pytest
from pygraphblas import INT64

from project.shortest_path import *
from tests.utils import read_graphs_from_json


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected",
    read_graphs_from_json(
        "test_sssp",
        lambda data: (
            data["I"],
            data["J"],
            data["V"],
            data["size"],
            data["start"],
            data["expected"],
        ),
        "graphs_shortest_path.json",
    ),
)
def test_sssp(I, J, V, size, start_vertex, expected):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    expected = [float(value) for value in expected]
    assert sssp(graph, start_vertex) == expected


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected",
    read_graphs_from_json(
        "test_mssp",
        lambda data: (
            data["I"],
            data["J"],
            data["V"],
            data["size"],
            data["start"],
            data["expected"],
        ),
        "graphs_shortest_path.json",
    ),
)
def test_mssp(I, J, V, size, start_vertex, expected):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    expected = [
        (start, [float(value) for value in dists]) for [start, dists] in expected
    ]
    assert mssp(graph, start_vertex) == expected


@pytest.mark.parametrize(
    "I, J, V, size, start_vertex, expected",
    read_graphs_from_json(
        "test_floyd_warshall",
        lambda data: (
            data["I"],
            data["J"],
            data["V"],
            data["size"],
            data["start"],
            data["expected"],
        ),
        "graphs_shortest_path.json",
    ),
)
def test_floyd_warshall(I, J, V, size, start_vertex, expected):
    graph = Matrix.from_lists(I, J, V, nrows=size, ncols=size)
    expected = [
        (start, [float(value) for value in dists]) for [start, dists] in expected
    ]
    assert floyd_warshall(graph) == expected


def test_non_square_adj_matrix():
    graph = Matrix.dense(FP64, 2, 3)
    with pytest.raises(ValueError):
        sssp(graph, 0)
        mssp(graph, [0])
        floyd_warshall(graph)


def test_unsupported_type_matrix():
    graph = Matrix.dense(INT64, 3, 3)
    with pytest.raises(ValueError):
        sssp(graph, 0)
        mssp(graph, [0])
        floyd_warshall(graph)
