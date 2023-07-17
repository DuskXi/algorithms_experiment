from typing import Callable

from matplotlib import pyplot as plt

from graph import Graph, TreeNode


def prim_algorithm(graph: Graph, callback: Callable = lambda node, edge, step: None) -> TreeNode:
    root = TreeNode(None, graph.nodes[0])
    visited = [root.data]
    queue = [root]
    i = 0
    while len(visited) < len(graph.nodes):
        for node in list(queue):
            neighbors = graph.neighbors_edges(node.data)
            if len(neighbors) == 0:
                queue.remove(node)
                continue
            min_edge = neighbors[0]
            for edge in neighbors:
                if edge[1] not in visited:
                    min_edge = min(min_edge, edge, key=lambda x: x[2])
            if min_edge[1] in visited:
                queue.remove(node)
                continue
            visited.append(min_edge[1])
            child = TreeNode(node, min_edge[1], weight=min_edge[2])
            node.add_child(child)
            queue.append(child)
            callback(child, min_edge, i)
            i += 1
    return root


def test_prim_algorithm():
    graph = Graph.random_graph(7, False, (1, 10), True)
    graph.draw()
    plt.show()
    root = prim_algorithm(graph)
    fig, ax = root.draw()
    plt.show()
