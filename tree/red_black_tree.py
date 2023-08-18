import networkx as nx
from matplotlib import pyplot as plt


class RedBlackNode:
    def __init__(self, value, color, parent=None, left=None, right=None):
        # self.key = key
        self.value = value
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

    def __str__(self):
        return f"{self.value} ({self.color})"

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = RedBlackNode(value, "red", self)
                if self.color == "red":
                    self.left.ab()

            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = RedBlackNode(value, "red", self)
                if self.color == "red":
                    self.right.ab()
            else:
                self.right.insert(value)

    def ab(self):
        parent = self.parent
        uncle, parent_direction = parent.get_brother()
        grandparent = parent.parent
        if uncle is not None and uncle.color == "red":
            parent.color = "black"
            uncle.color = "black"
            if grandparent.parent is not None:
                grandparent.color = "red"
                if grandparent.parent.color == "red":
                    grandparent.ab()
        else:
            if parent_direction == "left":
                if self == parent.right and self.color == "red":
                    parent.left = self.right
                    self.right = parent
                if grandparent.parent is not None:
                    grandparent.parent.left = parent
                parent.parent = grandparent.parent
                grandparent.parent = parent
                grandparent.left = parent.right
                parent.right = grandparent
                color = parent.color
                parent.color = grandparent.color
                grandparent.color = color
            else:
                if self == parent.left and self.color == "red":
                    parent.right = self.left
                    self.left = parent
                if grandparent.parent is not None:
                    grandparent.parent.right = parent
                parent.parent = grandparent.parent
                grandparent.parent = parent
                grandparent.right = parent.left
                parent.left = grandparent
                color = parent.color
                parent.color = grandparent.color
                grandparent.color = color

    def auto_balance(self, direction):
        bro, self_direction = self.get_brother()
        if bro is not None and bro.color == "red" and self.parent.parent is not None:
            self.color = "black"
            bro.color = "black"
            self.parent.color = "red"
            if self.parent.parent is not None:
                if self.parent.parent.color == "red":
                    self.draw()
                    plt.show()
                    self.parent.parent.auto_balance("left" if self.parent == self.parent.parent.left else "right")
                # self.parent.color = "red"
        else:
            if self_direction == "left":
                if direction == "right":
                    self.left = self.right
                    self.right = None
                self.parent.left = self.right
                self.right = self.parent
                gp = self.parent.parent
                self.parent.parent = self
                self.parent = gp
                if gp is not None:
                    gp.left = self
                color = self.color
                self.color = self.right.color
                self.right.color = color
            else:
                if direction == "left":
                    self.right = self.left
                    self.left = None
                self.parent.right = self.left
                self.left = self.parent
                gp = self.parent.parent
                self.parent.parent = self
                self.parent = gp
                if gp is not None:
                    gp.right = self
                color = self.color
                self.color = self.left.color
                self.left.color = color

    def get_brother(self):
        if self == self.parent.left:
            return self.parent.right, "left"
        else:
            return self.parent.left, "right"

    def find_root(self):
        if self.parent is None:
            return self
        return self.parent.find_root()

    def draw(self):
        if self.parent is not None:
            self.find_root().draw()
            return
        g = nx.Graph()
        self.add_to_graph(g)
        pos = nx.nx_pydot.pydot_layout(g, "dot")
        nx.draw(g, pos, with_labels=False, node_color=[n[1]["color"] for n in g.nodes(data=True)])
        nx.draw_networkx_labels(g, pos, font_color='white')

    def add_to_graph(self, g: nx.Graph):
        g.add_node(self.value, color=self.color)
        if self.left is not None:
            self.left.add_to_graph(g)
            g.add_edge(self.value, self.left.value)
        if self.right is not None:
            self.right.add_to_graph(g)
            g.add_edge(self.value, self.right.value)


if __name__ == "__main__":
    root = RedBlackNode(10, "black")
    root.left = RedBlackNode(6, "red", root)
    root.left.left = RedBlackNode(3, "black", root.left)
    root.left.right = RedBlackNode(8, "black", root.left)

    root.right = RedBlackNode(19, "black", root)
    root.right.left = RedBlackNode(12, "red", root.right)
    root.right.right = RedBlackNode(21, "red", root.right)

    root.insert(2)
    root.draw()
    plt.show()
    root.insert(1)
    root.draw()
    plt.show()
    root.insert(0)
    root.draw()
    plt.show()
    root.insert(11)
    root.draw()
    plt.show()
    root.insert(15)
    root.draw()
    plt.show()
    root = root.find_root()
    root.insert(16)
    root = root.find_root()
    root.draw()
    plt.show()
