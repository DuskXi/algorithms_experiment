from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from graph import Graph, TreeNode


def prim_algorithm(graph: Graph, callback: Callable = lambda node, edge, step: None, root_index=0) -> TreeNode:
    root = TreeNode(None, graph.nodes[root_index])
    visited = [root.data]
    queue = [(root, *x) for x in graph.neighbors_edges(root.data) if x[1] not in visited]
    i = 0
    while len(queue) > 0:
        min_edge = queue[0]
        for task in queue:
            edge = task[1:]
            if edge[2] < min_edge[3]:
                min_edge = task
        node = min_edge[0]
        edge = min_edge[1:]
        visited.append(edge[1])
        child = TreeNode(node, edge[1], weight=edge[2])
        node.add_child(child)
        queue += [(child, *x) for x in graph.neighbors_edges(child.data) if x[1] not in visited]
        queue = list(filter(lambda x: x[2] not in visited, queue))
        callback(child, edge, i)
        i += 1
    return root


def run_test():
    graph = Graph.random_graph(10, False, (1, 10), True)
    graph.draw(integer=True, alpha=0.35)
    plt.show()
    root = prim_algorithm(graph)
    fig, ax = root.draw()
    plt.title('Prim\'s Algorithm')
    plt.show()
    print()
    for line in graph.adjacency_matrix():
        for i, w in enumerate(line):
            if i == 0:
                print('[', end='')
            print(w, end='')
            if i == len(line) - 1:
                print('], ')
            else:
                print(', ', end='')
    print(root)


if __name__ == '__main__':
    run_test()
