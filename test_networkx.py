import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph import Graph

graph = Graph.random_graph(7, False, (1, 10), True)

fig, ax = graph.draw()
plt.show()

# # 创建一个networkx图对象
# G = nx.from_numpy_array(adj_matrix)
#
# # 绘制图
# pos = nx.spring_layout(G)  # 定义一个布局，这里使用了spring布局方式
# nx.draw(G, pos, with_labels=True)  # 绘制图形，with_labels决定节点是非带标签（编号），node_size是节点的直径，alpha是透明度
# labels = nx.get_edge_attributes(G, "weight")  # 获取边的权重
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # 在边上画出权重
# plt.show()  # 显示图形
