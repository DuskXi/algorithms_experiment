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
        self.node_table = {}
        self.directed = directed

    def merge(self, graph):
        self.nodes += graph.nodes
        self.edges += graph.edges

    def weight_less_edges(self):
        return list(map(lambda x: (x[0], x[1]), self.edges))

    def connected(self, node_1, node_2):
        # node_1 = self.node_table[str(node_1)] if type(node_1) != int else node_1
        # node_2 = self.node_table[str(node_2)] if type(node_2) != int else node_2
        return node_1 in self.neighbors(node_2)

    def neighbors(self, node):
        # node = self.node_table[str(node)] if type(node) != int else node
        neighbors = []
        for edge in self.edges:
            if edge[0] == node and edge[1] not in neighbors:
                neighbors.append(edge[1])
            elif edge[1] == node and edge[0] not in neighbors:
                neighbors.append(edge[0])
        return neighbors

    def neighbors_edges(self, node):
        # node = self.node_table[str(node)] if type(node) != int else node
        neighbors = []
        for edge in self.edges:
            if edge[0] == node and edge[1] not in neighbors:
                neighbors.append(edge)
            elif edge[1] == node and edge[0] not in neighbors:
                neighbors.append(edge)
        return neighbors

    def get_edge_weight(self, node_1, node_2):
        # node_1 = self.node_table[str(node_1)] if type(node_1) != int else node_1
        # node_2 = self.node_table[str(node_2)] if type(node_2) != int else node_2
        for edge in self.edges:
            if edge[0] == node_1 and edge[1] == node_2:
                return edge[2]
            elif self.directed and edge[0] == node_2 and edge[1] == node_1:
                return edge[2]
        return None

    def add_node(self, node):
        # if type(node) != int:
        #     node_str = str(node)
        #     node = len(self.nodes)
        # else:
        #     node_str = str(node)
        self.nodes.append(node)
        # self.node_table[node_str] = node
        self.edge_colors[node] = 'black'

    def add_edge(self, node_1, node_2, w):
        node_1 = self.nodes.index(str(node_1)) if type(node_1) != int else node_1
        node_2 = self.nodes.index(str(node_2)) if type(node_2) != int else node_2
        self.edges.append((node_1, node_2, w))
        if not self.directed:
            self.edges.append((node_2, node_1, w))

    def add_weight_less_edge(self, node_1, node_2):
        node_1 = self.nodes.index(str(node_1)) if type(node_1) != int else node_1
        node_2 = self.nodes.index(str(node_2)) if type(node_2) != int else node_2
        self.edges.append((node_1, node_2, 1))
        if not self.directed:
            self.edges.append((node_2, node_1, 1))

    def set_edge_weight(self, node_1, node_2, w):
        node_1 = self.nodes.index(str(node_1)) if type(node_1) != int else node_1
        node_2 = self.nodes.index(str(node_2)) if type(node_2) != int else node_2
        for i, edge in enumerate(self.edges):
            if edge[0] == node_1 and edge[1] == node_2:
                self.edges[i] = (node_1, node_2, w)
                return

        self.edges.append((node_1, node_2, w))

    def adjacency_list(self):
        adjacency_list = {}
        for node in self.nodes:
            node = self.nodes.index(str(node)) if type(node) != int else node
            adjacency_list[node] = self.neighbors(node)
        return adjacency_list

    def adjacency_matrix(self, dtype='float32', fill_with=None):
        adjacency_matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=dtype)
        adjacency_matrix.fill(fill_with)
        for edge in self.edges:
            adjacency_matrix[edge[0], edge[1]] = edge[2] if edge[2] != float('nan') and dtype == 'int' else 0
        return adjacency_matrix

    @staticmethod
    def from_adjacency_matrix(adjacency_matrix: np.ndarray, directed=False):
        graph = Graph(directed=directed)
        graph.nodes = list(range(len(adjacency_matrix)))
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if adjacency_matrix[i, j] != 0:
                    graph.edges.append((i, j, adjacency_matrix[i, j]))
        return graph

    def draw(self, fig=None, ax=None, alpha=0.5, integer=False, labels=None, layout=nx.spring_layout):
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for node in self.nodes:
            G.add_node(node)
        for edge in self.edges:
            G.add_edge(self.nodes[edge[0]], self.nodes[edge[1]], weight=edge[2])

        pos = layout(G)
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, with_labels=labels is None)
        if labels is not None:
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels if not integer else {k: str(int(v)) for k, v in labels.items()}, alpha=alpha)
        return fig, ax

    def draw_as_label_weight(self, fig=None, ax=None, alpha=0.5):
        adjacency_matrix = self.adjacency_matrix('object')
        G = nx.DiGraph() if self.directed else nx.MultiGraph()
        # for i in range(adjacency_matrix.shape[0]):
        #     for j in range(adjacency_matrix.shape[1]):
        #         if adjacency_matrix[j, i] is not None:
        #             G.add_edge(self.nodes[i], self.nodes[j], label=adjacency_matrix[i, j])
        for edge in self.edges:
            G.add_edge(self.nodes[edge[0]], self.nodes[edge[1]], label=edge[2])
        pos = nx.spring_layout(G)
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        # nx.draw(G, pos, ax=ax, with_labels=True, connectionstyle='arc3, rad=0.2')

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600)
        nx.draw_networkx_labels(G, pos, ax=ax)
        labels = nx.get_edge_attributes(G, "label")
        edge_labels_local = {}
        for edge in G.edges(data=True):
            if (edge[1], edge[0]) in G.edges() and edge[0] != edge[1]:  # Check if the reverse edge exists
                # This is a bidirectional edge
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(edge[0], edge[1])], connectionstyle="arc3,rad=0.2", edge_color='k', arrowstyle='-|>', arrowsize=20)
            else:
                # This is a unidirectional edge
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(edge[0], edge[1])], edge_color='k', arrowstyle='-|>', arrowsize=20)
            if edge[0] == edge[1]:
                edge_labels_local[(edge[0], edge[1])] = labels[(edge[0], edge[1])]
                del labels[(edge[0], edge[1])]

        my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels, alpha=alpha, rotate=False, rad=0.2)
        my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels_local, alpha=alpha, rotate=False, rad=0.2, vertical_offset=0.2)

        return fig, ax

    def draw_as_flow_network(self, flow: np.ndarray, fig=None, ax=None, s=-1, t=-1, alpha=0.5, integer=True, kamada=True, f=True, pos=None):
        adjacency_matrix = self.adjacency_matrix(dtype='int', fill_with=0)
        if s == -1 or s not in self.nodes:
            s = self.nodes[0]
        if t == -1 or t not in self.nodes:
            t = self.nodes[-1]
        G = nx.MultiDiGraph()

        # 添加边和它们的属性
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] != 0:
                    G.add_edge(i, j, capacity=adjacency_matrix[i, j], flow=flow[i, j])
                if adjacency_matrix[j, i] != 0:
                    G.add_edge(j, i, capacity=adjacency_matrix[j, i], flow=flow[j, i])
        if pos is None:
            if kamada:
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spectral_layout(G)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600)
        for edge in G.edges(data=True):
            if (edge[1], edge[0]) in G.edges():  # Check if the reverse edge exists
                # This is a bidirectional edge
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(edge[0], edge[1])], connectionstyle="arc3,rad=0.2", edge_color='k', arrowstyle='-|>', arrowsize=20)
            else:
                # This is a unidirectional edge
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(edge[0], edge[1])], edge_color='k', arrowstyle='-|>', arrowsize=20)

        if f:
            edge_labels_forward = {(edge[0], edge[1]): f"{int(edge[2]['flow'])}/{int(edge[2]['capacity'])}" for edge in G.edges(data=True) if edge[0] < edge[1]}
            edge_labels_backward = {(edge[0], edge[1]): f"{int(edge[2]['flow'])}/{int(edge[2]['capacity'])}" for edge in G.edges(data=True) if edge[0] > edge[1]}
        else:
            edge_labels_forward = {(edge[0], edge[1]): f"{int(edge[2]['capacity'])}" for edge in G.edges(data=True) if edge[0] < edge[1]}
            edge_labels_backward = {(edge[0], edge[1]): f"{int(edge[2]['capacity'])}" for edge in G.edges(data=True) if edge[0] > edge[1]}

        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels_forward, verticalalignment='bottom')
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels_backward, verticalalignment='top')
        # 创建源节点和目标节点的标签
        node_labels = {s: 's', t: 't'}

        # 绘制源节点和目标节点的标签
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='red')

        other_nodes = set(G.nodes()) - {s, t}
        other_labels = {node: node for node in other_nodes}
        nx.draw_networkx_labels(G, pos, labels=other_labels)
        # edge_labels = nx.get_edge_attributes(G, 'capacity')
        # nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
        return fig, ax, pos

    def set_edge_color(self, edge_color):
        self.edge_colors = edge_color

    def get_edge_color_list(self):
        # return sort by edge array
        return [self.edge_colors[edge] for edge in self.weight_less_edges()]

    def draw_graphviz(self, fig=None, ax=None, alpha=0.5, integer=False):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        G = nx.from_numpy_array(self.adjacency_matrix(), create_using=nx.DiGraph)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, ax=ax, with_labels=True)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels if not integer else {k: str(int(v)) for k, v in labels.items()}, alpha=alpha)
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
        adj_matrix = np.array(adj_matrix, dtype=np.int)
        graph = Graph(directed=directed)
        for i in range(num_nodes):
            graph.add_node(i)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    graph.add_edge(i, j, adj_matrix[i, j])

        return graph


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

    def to_graph(self, enable_con=False):
        graph = Graph(True)
        graph.add_node(self.data)
        self.add_to_graph(graph, enable_con=enable_con)
        return graph

    def add_to_graph(self, graph, con=0, enable_con=False):
        for child in self.children:
            graph.add_node(child.data)
            graph.add_edge(graph.nodes.index(self.data), graph.nodes.index(child.data), con + (1 if not child.weight else child.weight))
            child.add_to_graph(graph, con + (1 if not child.weight else child.weight) if enable_con else 0, enable_con)

    # def draw(self, fig=None, ax=None, enable_con=True, show_weight=True):
    #     graph = self.to_graph(enable_con=enable_con)
    #     G = nx.from_numpy_array(graph.adjacency_matrix(fill_with=0))
    #     # add node label to graph
    #     labels = {}
    #     for i in range(len(graph.nodes)):
    #         labels[i] = graph.nodes[i]
    #     nx.set_node_attributes(G, labels, "label")
    #     labels = nx.get_node_attributes(G, 'label')
    #     pos = graphviz_layout(G, prog="dot")
    #     if fig is None or ax is None:
    #         fig, ax = plt.subplots()
    #     nx.draw(G, pos, ax=ax, with_labels=True, labels=labels)
    #     if show_weight:
    #         labels = nx.get_edge_attributes(G, "weight")
    #         # draw with int
    #         nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={k: str(int(v)) for k, v in labels.items()})
    #     return fig, ax

    def draw(self, fig=None, ax=None, enable_con=True, show_weight=True):
        graph = self.to_graph(enable_con=enable_con)
        G = nx.DiGraph()
        for node in graph.nodes:
            G.add_node(node)
        for edge in graph.edges:
            G.add_edge(graph.nodes[edge[0]], graph.nodes[edge[1]], weight=edge[2])
        pos = graphviz_layout(G, prog="dot")
        scale_factor = 2.0
        # for node in pos:
        #     pos[node] = (pos[node][0] + scale_factor, pos[node][1])

        # pos = nx.spring_layout(G)
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=600, font_size=7)
        # 使用matplotlib的scatter函数绘制节点
        # 使用matplotlib的plot函数绘制边
        # for edge in G.edges():
        #     x1, y1 = pos[edge[0]]
        #     x2, y2 = pos[edge[1]]
        #     ax.plot([x1, x2], [y1, y2], 'k-')
        #
        # for node, (x, y) in pos.items():
        #     ax.scatter(x, y, s=600, color='lightblue')  # s参数控
        #     ax.text(x, y, node, ha='center', va='center')
        #
        # plt.axis('off')

        if show_weight:
            labels = nx.get_edge_attributes(G, "weight")
            # draw with int
            nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={k: str(int(v)) for k, v in labels.items()})
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


def my_draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=None,
        label_pos=0.5,
        font_size=10,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        rotate=True,
        clip_on=True,
        rad=0.0,
        vertical_offset=0.0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    :param rad:
    :param vertical_offset:
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)
        y += vertical_offset

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
