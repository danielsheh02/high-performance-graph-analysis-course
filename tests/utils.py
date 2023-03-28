import json
import pathlib
import inspect


def read_graphs_from_json(name, conf, file_name):
    with pathlib.Path(inspect.stack()[1].filename) as f:
        parent = f.parent
    with open(parent / "graphs" / file_name, "r") as file:
        data = json.load(file)
        return [conf(block) for block in data[name]]
