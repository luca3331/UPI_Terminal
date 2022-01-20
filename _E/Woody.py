import numpy as np


class Tree:
    """
    nX:子ノードへのポインタ
    data:合致度
    tri_arr:状態行列

    """

    def __init__(self, data):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        self.n4 = None
        self.n5 = None
        self.n6 = None
        self.n7 = None
        self.n8 = None
        self.n9 = None
        self.n10 = None
        self.n11 = None
        self.n12 = None
        self.n13 = None
        self.n14 = None
        self.n15 = None
        self.n16 = None
        self.n17 = None
        self.n18 = None
        self.n19 = None
        self.n20 = None
        self.n21 = None
        self.n22 = None
        self.data = data
        self.tri_list = np.full((24, 24), -1)


root = Tree(100)
root.n1 = Tree(10)
root.n1.n1 = Tree(1)

print(root.n1.n1.data)
