import networkx as nx
from matplotlib import pyplot as plt

from graph import Graph

adj_list = [
    ("v1", "v8"),
    ("v2", "v1"),
    ("v2", "v7"),
    ("v2", "v4"),
    ("v3", "v2"),
    ("v4", "v3"),
    ("v4", "v6"),
    ("v5", "v4"),
    ("v6", "v5"),
    ("v6", "v2"),
    ("v7", "v6"),
    ("v7", "v2"),
    ("v8", "v2"),
    ("v8", "v7"),
]

graph = Graph(True)
for i in range(1, 9):
    graph.add_node(f"v{i}")

for e in adj_list:
    graph.add_edge(e[0], e[1], 1)

if __name__ == "__main__":
    print(graph.adjacency_matrix(fill_with=0))
    print(graph.adjacency_list())
    graph.draw(layout=nx.circular_layout)
    plt.show()
