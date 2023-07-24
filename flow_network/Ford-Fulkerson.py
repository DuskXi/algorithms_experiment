import matplotlib.pyplot as plt
import numpy as np

from graph import Graph


def ford_fulkerson(graph: Graph, s: int, t: int, pos):
    g = np.array(graph.nodes)
    resnet = Graph.from_adjacency_matrix(np.array(graph.adjacency_matrix()), True)
    resnet_independent = Graph.from_adjacency_matrix(np.array(graph.adjacency_matrix()), True)
    while dfs(resnet, s, t) is not None:
        path = dfs(resnet, s, t)
        min_flow = min([edge[2] for edge in path])
        apath = argument_path(resnet, path)
        for edge, a_edge in zip(path, apath):
            resnet.set_edge_weight(edge[0], edge[1], edge[2] - min_flow)
            resnet.set_edge_weight(a_edge[0], a_edge[1], a_edge[2] + min_flow)
            resnet_independent.set_edge_weight(edge[0], edge[1], edge[2] - min_flow)
        print_resnet(resnet, pos)

    return Graph.from_adjacency_matrix(np.array(graph.adjacency_matrix() - resnet_independent.adjacency_matrix()), True)


def dfs(g: Graph, current, target, visited=[]):
    for edge in [x for x in g.neighbors_edges(current) if x[0] == current and x[2] > 0 and x[1] not in visited]:
        if edge[1] == target:
            return [edge]
        else:
            result = dfs(g, edge[1], target, visited + [current])
            if result is not None:
                return [edge] + result
    return None


def argument_path(g: Graph, path):
    apath = []
    for p in path:
        neighbors = g.neighbors_edges(p[1])
        w = 0
        for n in neighbors:
            if n[1] == p[0]:
                w = n[2]
                break
        apath.append((p[1], p[0], w))
    return apath


def print_resnet(graph: Graph, pos):
    fig, ax, pos = graph.draw_as_flow_network(np.zeros((len(graph.nodes), len(graph.nodes))), f=False, pos=pos)
    plt.show()


def run_test():
    # graph = Graph.random_graph(7, True, (1, 10), True)
    adj_matrix_7 = np.array([
        [0, 1, 3, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 4, 0, 0],
        [0, 3, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    graph = Graph.from_adjacency_matrix(adj_matrix_7, True)
    fig, ax, pos = graph.draw_as_flow_network(np.zeros((len(graph.nodes), len(graph.nodes))))
    plt.show()

    gf = ford_fulkerson(graph, 0, 8, pos)

    graph.draw_as_flow_network(gf.adjacency_matrix())
    plt.title('Ford-Fulkerson')
    plt.show()


if __name__ == '__main__':
    run_test()
