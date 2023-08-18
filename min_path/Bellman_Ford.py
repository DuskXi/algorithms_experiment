import copy

import networkx as nx
from matplotlib import pyplot as plt

from graph import Graph


def bellman_ford(graph: Graph, start: str, init=None, p=False, include=False, single_sink=False, max_inter=-1):
    start = graph.nodes.index(start)
    distance = [float('inf') for node in graph.nodes]
    if init is not None and type(init) == list and len(init) == len(graph.nodes) and init[start] == 0:
        distance = init
    distance[start] = 0
    last = copy.deepcopy(distance)
    for i in range(len(graph.nodes) - 1):
        if p:
            print("".join([f"{n}: {d}\t" for n, d in zip(graph.nodes, distance)]))
        if 0 < max_inter <= i:
            break
        for node in graph.nodes:
            node = graph.nodes.index(node)
            if node == start:
                continue
            edges = graph.neighbors_edges(node)
            values = []
            if include:
                values.append(distance[node])
            for edge in edges:
                if edge[0 if single_sink else 1] == node:
                    values.append(distance[edge[1 if single_sink else 0]] + edge[2])
            if len(values) > 0:
                distance[node] = min(values)
        if last == distance:
            break
        last = copy.deepcopy(distance)
    # for edge in graph.edges:
    #     if distance[edge[1]] > distance[edge[0]] + edge[2]:
    #         return False
    return distance


def data():
    graph = Graph(True)
    for i in range(1, 11):
        graph.add_node(f"v{i}")
    edges = [
        ("v1", "v5", 5),
        ("v2", "v1", 7),
        ("v2", "v7", -4),
        ("v3", "v2", 5),
        ("v4", "v3", -2),
        ("v5", "v4", -4),
        ("v5", "v10", -6),
        ("v6", "v1", 8),
        ("v6", "v8", -1),
        ("v7", "v9", 4),
        ("v8", "v3", -2),
        ("v8", "v10", -2),
        ("v9", "v4", -3),
        ("v9", "v6", 7),
    ]
    for e in edges:
        graph.add_edge(e[0], e[1], e[2])
    return graph


if __name__ == '__main__':
    g = data()
    g.draw(layout=nx.nx_pydot.graphviz_layout)
    plt.show()
    result = bellman_ford(g, "v3", p=True, init=[-1, 6, 0, -2, -6, 7, -1, -2, -5, -3], single_sink=True, max_inter=2)
    print(result)
    labels = {}
    for i, dis in enumerate(result):
        labels[g.nodes[i]] = f"{g.nodes[i]}: {dis}"
    g.draw(layout=nx.nx_pydot.graphviz_layout, labels=labels)
    plt.show()
