import random
from typing import Callable

import numpy as np
import pydot
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, directed=False):
        # [0,1,2,3] or ['A','B','C','D']
        self.edge_colors = {}
        self.nodes = []
        # [(node_1,node_2, w)]
        self.edges = []
        self.directed = directed

    def weight_less_edges(self):
        return list(map(lambda x: (x[0], x[1]), self.edges))

    def connected(self, node_1, node_2):
        return node_1 in self.neighbors(node_2)

    def neighbors(self, node):
        neighbors = []
        for edge in self.edges:
            if edge[0] == node and edge[1] not in neighbors:
                neighbors.append(edge[1])
            elif edge[1] == node and edge[0] not in neighbors:
                neighbors.append(edge[0])
        return neighbors

    def neighbors_edges(self, node):
        neighbors = []
        for edge in self.edges:
            if edge[0] == node and edge[1] not in neighbors:
                neighbors.append(edge)
            elif edge[1] == node and edge[0] not in neighbors:
                neighbors.append(edge)
        return neighbors

    def get_edge_weight(self, node_1, node_2):
        for edge in self.edges:
            if edge[0] == node_1 and edge[1] == node_2:
                return edge[2]
            elif self.directed and edge[0] == node_2 and edge[1] == node_1:
                return edge[2]
        return None

    def add_node(self, node):
        self.nodes.append(node)
        self.edge_colors[node] = 'black'

    def add_edge(self, node_1, node_2, w):
        self.edges.append((node_1, node_2, w))
        if not self.directed:
            self.edges.append((node_2, node_1, w))

    def add_weight_less_edge(self, node_1, node_2):
        self.edges.append((node_1, node_2, 1))
        if not self.directed:
            self.edges.append((node_2, node_1, 1))

    def adjacency_list(self):
        adjacency_list = {}
        for node in self.nodes:
            adjacency_list[node] = self.neighbors(node)
        return adjacency_list

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            adjacency_matrix[edge[0], edge[1]] = edge[2]
        return adjacency_matrix

    def draw(self, fig=None, ax=None):
        if self.directed:
            G = nx.from_numpy_array(self.adjacency_matrix(), create_using=nx.DiGraph)
        else:
            G = nx.from_numpy_array(self.adjacency_matrix())
        pos = nx.spring_layout(G)
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, with_labels=True)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels)
        return fig, ax

    def set_edge_color(self, edge_color):
        self.edge_colors = edge_color

    def get_edge_color_list(self):
        # return sort by edge array
        return [self.edge_colors[edge] for edge in self.weight_less_edges()]

    def draw_graphviz(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        G = nx.from_numpy_array(self.adjacency_matrix(), create_using=nx.DiGraph)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, ax=ax, with_labels=True)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels)
        return fig, ax

    @staticmethod
    def random_graph(num_nodes=5, directed=False, value_range=(1, 10), random_zero=False):
        # 生成一个随机的邻接矩阵，值在1-10之间
        adj_matrix = np.random.randint(*value_range, size=(num_nodes, num_nodes))
        # 为了确保图是无向的，我们将矩阵设置为对称的
        if not directed:
            adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
        # 将回环设置为0
        np.fill_diagonal(adj_matrix, 0)
        # 如果random_zero为True，则随机将一些值设置为0
        if random_zero:
            adj_matrix = np.where(np.random.randint(0, 2, size=(num_nodes, num_nodes)), adj_matrix, 0)
        adj_matrix = np.array(adj_matrix, dtype=np.int64)
        graph = Graph(directed=directed)
        for i in range(num_nodes):
            graph.add_node(i)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    graph.add_edge(i, j, adj_matrix[i, j])

        return graph


class Path:
    def __init__(self):
        self.nodes = []


class TreeNode:
    def __init__(self, parent, data, weight=None, __str__: Callable = None):
        self.data = data
        self.weight = weight
        self.children = []
        self.parent = parent
        self.__str__ = __str__

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def __repr__(self):
        if self.__str__ is not None:
            return self.__str__(self.data)
        else:
            return str(self.data)

    def prune(self):
        if self.parent is not None:
            if self in self.parent.children:
                self.parent.children.remove(self)
            self.parent = None
            for child in self.children:
                child.prune()

    def to_graph(self):
        graph = Graph()
        graph.add_node(self.data)
        self.add_to_graph(graph)
        return graph

    def add_to_graph(self, graph):
        for child in self.children:
            graph.add_node(child.data)
            graph.add_edge(graph.nodes.index(self.data), graph.nodes.index(child.data), 1 if not child.weight else self.weight)
            child.add_to_graph(graph)

    def draw(self, fig=None, ax=None):
        graph = self.to_graph()
        G = nx.from_numpy_array(graph.adjacency_matrix())
        # add node label to graph
        labels = {}
        for i in range(len(graph.nodes)):
            labels[i] = graph.nodes[i]
        nx.set_node_attributes(G, labels, "label")
        labels = nx.get_node_attributes(G, 'label')
        pos = graphviz_layout(G, prog="dot")
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, with_labels=True, labels=labels)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels)
        return fig, ax

    @staticmethod
    def generate_random_tree(depth):
        used_data = set()

        def _generate_node(parent, _depth):
            if _depth > 0:
                if len(used_data) == 100:
                    return None
                while data := random.randint(0, 100):
                    if data not in used_data:
                        break
                used_data.add(data)

                node = TreeNode(parent, data)
                for _ in range(random.randint(2, 4)):  # 生成2到4个子节点
                    child = _generate_node(node, _depth - 1)
                    if child:
                        node.add_child(child)
                return node
            else:
                return None

        return _generate_node(None, depth)


class TreeGraph(Graph):
    def __init__(self):
        super().__init__(False)
        root = 0

    # noinspection PyMethodOverriding
    def add_node(self, parent, node):
        self.nodes.append(node)
        self.edges.append((parent, node, 1))

    def delete_node(self, node):
        self.nodes.remove(node)
        for edge in self.edges:
            if edge[0] == node or edge[1] == node:
                self.edges.remove(edge)
