import copy
import time
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from graph import Graph, TreeNode


def union(pool, v1, v2, w):
    root1 = find_root(pool, v1)
    root2 = find_root(pool, v2)
    if root1 is None or root2 is None:
        return False
    if root1 == root2:
        return False
    root1.add_child(root2)
    root2.parent = root1
    root2.weight = w
    return True


def find_root(pool: list[TreeNode], v) -> TreeNode:
    for node in pool:
        if node.data == v:
            return find_parent(node)


# def find_parent(node, init=True):
#     if node.parent is None:
#         return [node] if not init else node
#     result = find_parent(node.parent, False)
#     if not init:
#         return [*result, node]
#     else:
#         root = result[0]
#         for n in result[1:]:
#             n.parent = root
#         return root


def find_parent(node):
    if node.parent is None:
        return node
    if node.parent.parent is not None:
        node.parent = node.parent.parent
    return find_parent(node.parent)


def get_roots(pool):
    roots = []
    for node in pool:
        root = find_root(pool, node.data)
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def kruskal_algorithm(graph: Graph, callback: Callable = lambda node, edge, step: None) -> TreeNode:
    unconnected = [TreeNode(None, node) for node in graph.nodes]
    edges = [(unconnected[x[0]], unconnected[x[1]], x[2]) for x in graph.edges]
    edges.sort(key=lambda x: x[2])
    banned = []
    for edge in edges:
        node1, node2, w = edge
        if node1.data in banned or node2.data in banned:
            continue
        status = union(unconnected, node1.data, node2.data, w)
        # banned.append(node1.data)
        callback(node1, edge, 0)
    roots = get_roots(unconnected)
    return roots[0]


def run_test():
    print("Running")
    graph = Graph.random_graph(20, False, (1, 10), True)
    graph.draw(integer=True, alpha=0.35)
    plt.show()
    start = time.time()
    root = kruskal_algorithm(copy.deepcopy(graph))
    end = time.time()
    fig, ax = root.draw(enable_con=False)
    plt.title(f'Kruskal Algorithm, Time: {1000 * (end - start):.2f}ms')
    plt.show()
    print("Time: Kruskal Algorithm: ", 1000 * (end - start), 'ms')
    from graph_algorithm.prim import prim_algorithm
    start = time.time()
    root = prim_algorithm(copy.deepcopy(graph), root_index=graph.nodes.index(root.data))
    end = time.time()
    fig, ax = root.draw(enable_con=False)
    plt.title(f'Prim Algorithm, Time: {1000 * (end - start):.2f}ms')
    plt.show()
    print("Time: Prim Algorithm: ", 1000 * (end - start), 'ms')


if __name__ == '__main__':
    run_test()
