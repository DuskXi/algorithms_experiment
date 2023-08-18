from typing import Callable, Union

from matplotlib import pyplot as plt

from graph import TreeNode
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300


class Tree234:
    def __init__(self, parent=None, data=None, __str__: Callable = None):
        self.parent = parent
        self.data = data
        self.children = []
        self.__str__ = __str__

    def __repr__(self):
        if self.__str__ is not None:
            return self.__str__(self.data)
        else:
            return str(self.data)

    def is_leaf(self):
        return len(self.children) == 0

    def find(self, value, give_near=False):
        if len(self.children) == 0:
            if value == self.data[0]:
                return self
            else:
                return None if not give_near else (self,)
        else:
            last = self.data[0]
            if value < self.data[0]:
                return self.children[0].find(value, give_near)
            elif value > self.data[-1]:
                return self.children[-1].find(value, give_near)
            for i, data in enumerate(self.data):
                if value == data:
                    return self
                elif last < value < data:
                    return self.children[i].find(value, give_near)
                last = data

    def find_with_fuse(self, value):
        # 删除过程中向下寻找时遇到2节点则合并
        if len(self.children) == 0:
            if value in self.data:
                return self
            else:
                return None
        else:
            if len(self.data) == 1 and self.parent is not None:
                self = self.fuse()
            # If the root is a 2-node,
            # and its children are 2-nodes,
            # fuse it with its children
            # if len(self.data) > 1:
            #     # check is it all child are 2 node
            #     is_2 = True
            #     for child in self.children:
            #         if len(child.data) != 1:
            #             is_2 = False
            #             break
            #     if is_2:
            #         # fuse
            #         self.children[-1].fuse()
            last = self.data[0]
            if value < self.data[0]:
                return self.children[0].find_with_fuse(value)
            elif value > self.data[-1]:
                return self.children[-1].find_with_fuse(value)
            for i, data in enumerate(self.data):
                if value == data:
                    return self
                elif last < value < data:
                    return self.children[i].find_with_fuse(value)
                last = data

    def find_replace(self, direct="left", depth=0):
        i = 0 if direct == "left" else -1
        if len(self.children) == 0:
            return self, self.data[i]
        if depth == 0:
            direct = "left" if direct == "right" else "right"
        return self.children[i].find_replace(direct, depth + 1)

    def fuse(self):
        if len(self.parent.data) > 1:
            brother, direct, brother_index = self.brother()
            if brother is not None:
                if len(brother.data) == 1:
                    # 融合
                    if direct == 'left':
                        self.data = brother.data + [self.parent.data.pop(-1)] + self.data
                        self.children = brother.children + self.children
                        self.parent.children.remove(brother)
                        brother.parent = None
                        brother.data = None
                        brother.children = None
                        return self
                    else:
                        self.data = self.data + [self.parent.data.pop(0)] + brother.data
                        self.children = self.children + brother.children
                        self.parent.children.remove(brother)
                        brother.parent = None
                        brother.data = None
                        brother.children = None
                        return self
        elif len(self.parent.data) == 1:
            brother, direct, brother_index = self.brother()
            if brother is not None:
                if len(brother.data) == 1:
                    if direct == 'left':
                        self.parent.data = brother.data + [self.parent.data[0]] + self.data
                        self.parent.children = brother.children + self.children
                        for child in self.parent.children:
                            child.parent = self.parent
                        brother.parent = None
                        brother.data = None
                        brother.children = None
                        return self.parent
                    else:
                        self.parent.data = self.data + [self.parent.data[0]] + brother.data
                        self.parent.children = self.children + brother.children
                        for child in self.parent.children:
                            child.parent = self.parent
                        brother.parent = None
                        brother.data = None
                        brother.children = None
                        return self.parent

    def insert(self, value):
        if len(self.children) == 0:
            if len(self.data) < 3:
                self.data.append(value)
                self.data.sort()
                return self.root(), True
            else:
                return self.split_insert().insert(value)
        else:
            result = self.find(value, give_near=True)
            if type(result) != tuple:
                return self.root(), False
            else:
                return result[0].insert(value)

    def clean(self):
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)
        self.parent = None
        self.data = None
        self.children = None

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    def split_insert(self):
        if len(self.data) != 3:
            raise ValueError('Cannot split a node with less than 3 children')
        mid = self.data[1]
        if self.parent is None:
            self.parent = Tree234(None, [])
            self.parent.children = [self]
        elif len(self.parent.data) == 3:
            self.parent = self.parent.split_insert()
        self.parent.data.append(mid)
        self.parent.data.sort()
        i = self.parent.children.index(self)
        self.parent.children.remove(self)
        left = Tree234(self.parent, [self.data[0]])
        left.children = self.children[:2] if len(self.children) > 0 else []
        right = Tree234(self.parent, [self.data[2]])
        right.children = self.children[2:] if len(self.children) > 0 else []
        self.parent.children.insert(i, left)
        self.parent.children.insert(i + 1, right)
        self.clean()
        return right

    def brother(self):
        index = self.parent.children.index(self)
        direct = "right" if index == 0 else "left"
        if direct == "left":
            if index > 0:
                return self.parent.children[index - 1], direct, index - 1
        else:
            if index < len(self.parent.children) - 1:
                return self.parent.children[index + 1], direct, index + 1
        return None, direct, index

    def delete(self, value):
        node = self.find_with_fuse(value)
        if node is None:
            return False
        else:
            if node.is_leaf():
                if len(node.data) == 1:
                    brother, direct, brother_index = node.brother()
                    if brother is not None:
                        bro_ch_i = -1 if direct == "left" else 0
                        data_i_in_parent = brother_index if direct == "left" else brother_index - 1
                        if len(brother.data) > 1:
                            if brother.is_leaf():
                                node.data[0] = node.parent.data[data_i_in_parent]
                                node.parent.data[data_i_in_parent] = brother.data[bro_ch_i]
                                brother.data.pop(bro_ch_i)
                            else:
                                pass
                        else:
                            if brother.is_leaf():
                                brother.data.insert(bro_ch_i, node.parent.data.pop(data_i_in_parent)) if bro_ch_i == 0 else brother.data.append(node.parent.data.pop(data_i_in_parent))
                                node.parent.children.remove(node)
                            else:
                                pass
                else:
                    node.data.remove(value)
            else:
                index = node.data.index(value)
                direct = "left"
                if len(node.data) > 1:
                    if index + 1 <= len(node.data) / 2:
                        direct = "left"
                    elif index + 1 >= len(node.data) / 2:
                        direct = "right"
                replace, v = node.find_replace(direct)
                node.data[index] = v
                replace.delete(v)

    def to_tree(self, parent=None):
        node = TreeNode(parent, str(self.data))
        for child in self.children:
            node.add_child(child.to_tree(parent))
        return node

    def draw(self):
        fig, ax = self.to_tree().draw(enable_con=False, show_weight=False)
        # set text on left top
        title = f' (balanced:{self.is_balanced()})'
        return fig, ax, title

    def is_balanced(self):
        is_leaf = self.is_leaf()
        for i, v in enumerate(self.data):
            if i < len(self.data) - 1:
                if v > self.data[i + 1]:
                    return False
            if not is_leaf:
                left = self.children[i]
                right = self.children[i + 1]
                if left.data[-1] > v or right.data[0] < v:
                    return False
        for child in self.children:
            if not child.is_balanced():
                return False
        return True


def test_exam():
    tree = Tree234(data=[9, 13, 17])

    node57 = Tree234(tree, data=[5, 7])
    node123 = Tree234(node57, data=[1, 2, 3])
    node6 = Tree234(node57, data=[6])
    node8 = Tree234(node57, data=[8])
    node57.children = [node123, node6, node8]

    node11 = Tree234(tree, data=[11])
    node10 = Tree234(node11, data=[10])
    node12 = Tree234(node11, data=[12])
    node11.children = [node10, node12]

    node15 = Tree234(tree, data=[15])
    node14 = Tree234(node15, data=[14])
    node16 = Tree234(node15, data=[16])
    node15.children = [node14, node16]

    node1912 = Tree234(tree, data=[19, 21])
    node18 = Tree234(node1912, data=[18])
    node20 = Tree234(node1912, data=[20])
    node222324 = Tree234(node1912, data=[22, 23, 24])
    node1912.children = [node18, node20, node222324]

    tree.children = [node57, node11, node15, node1912]
    # tree.children = [node257]

    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree' + title)
    plt.show()

    tree.insert(4)

    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree' + title)
    plt.show()

    tree.delete(15)

    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree' + title)
    plt.show()



def test_fuse_delete():
    tree = Tree234(data=[4, 8])
    tree.children = [Tree234(tree, [2]), Tree234(tree, [6]), Tree234(tree, [10])]
    tree.children[0].children = [Tree234(tree.children[0], [1]), Tree234(tree.children[0], [3])]
    tree.children[1].children = [Tree234(tree.children[1], [5]), Tree234(tree.children[1], [7])]
    tree.children[2].children = [Tree234(tree.children[2], [9]), Tree234(tree.children[2], [11, 12])]

    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree' + title)
    plt.show()

    # tree.delete(12)
    #
    # fig, ax, title = tree.draw()
    # plt.title('2-3-4 Tree' + title)
    # plt.show()
    #
    # tree.delete(11)
    #
    # fig, ax, title = tree.draw()
    # plt.title('2-3-4 Tree' + title)
    # plt.show()
    #
    # tree.delete(10)
    # tree.delete(9)
    #
    # fig, ax, title = tree.draw()
    # plt.title('2-3-4 Tree' + title)
    # plt.show()
    #
    # tree.delete(8)
    # fig, ax, title = tree.draw()
    # plt.title('2-3-4 Tree' + title)
    # plt.show()
    # tree.delete(6)
    # fig, ax, title = tree.draw()
    # plt.title('2-3-4 Tree' + title)
    # plt.show()

    tree.delete(8)

    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree' + title)
    plt.show()


def run_test():
    tree = Tree234(data=[8])
    tree.children = [Tree234(tree, [3, 5]), Tree234(tree, [10, 13])]
    tree.children[0].children = [Tree234(tree.children[0], [2]), Tree234(tree.children[0], [4]), Tree234(tree.children[0], [6])]
    tree.children[1].children = [Tree234(tree.children[1], [9]), Tree234(tree.children[1], [11, 12]), Tree234(tree.children[1], [16])]
    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree' + title)
    plt.show()

    print(tree.find(3))

    # add

    # tree, result = tree.insert(20)
    # fig, ax, title = tree.draw()
    # plt.title('2-3-4 Tree' + title)
    # plt.show()
    #
    # for i in range(21, 40):
    #     tree, result = tree.insert(i)
    # fig, ax, title = tree.draw()
    # fig.set_size_inches(10, 8)
    # plt.title('2-3-4 Tree +21-40' + title)
    # plt.show()

    # delete

    tree.delete(16)
    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree delete 16' + title)
    plt.show()

    tree.delete(13)
    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree delete 13' + title)
    plt.show()

    tree.delete(9)
    fig, ax, title = tree.draw()
    plt.title('2-3-4 Tree delete 9' + title)
    plt.show()


if __name__ == '__main__':
    # run_test()
    # test_fuse_delete()
    # test_exam()
    test_exam()
