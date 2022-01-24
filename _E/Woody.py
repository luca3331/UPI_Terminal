import copy

import numpy as np


class Tree:
    """
    nX:子ノードへのポインタ
    score:合致度
    tri_arr:状態行列

    """

    def __init__(self):
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
        self.move = None
        self.score = -1
        self.tri_list = np.full((24, 24), -1)

    def init_2(self):
        self.n1 = Tree()
        self.n2 = Tree()
        self.n3 = Tree()
        self.n4 = Tree()
        self.n5 = Tree()
        self.n6 = Tree()
        self.n7 = Tree()
        self.n8 = Tree()
        self.n9 = Tree()
        self.n10 = Tree()
        self.n11 = Tree()
        self.n12 = Tree()
        self.n13 = Tree()
        self.n14 = Tree()
        self.n15 = Tree()
        self.n16 = Tree()
        self.n17 = Tree()
        self.n18 = Tree()
        self.n19 = Tree()
        self.n20 = Tree()
        self.n21 = Tree()
        self.n22 = Tree()

    def init_3(self):
        for num in range(1, 23):
            name = 'n' + str(num)
            setattr(self, name, Tree())

    def tree_proc(self, move, depth, pos1, root_count):
        """
        if depth != 0:
            depthが0でなければ現在のノードからさらに下にノードを増やす．
            0であれば現在のノードが葉ノードになる．
        """
        if pos1.field.is_death():
            return -999999, Move.none()
        moves = generate_moves(pos1, positions_common.tumo_pool)
        if depth == 2:
            self.score, self.tri_list = proc()
            self.move = move

        for ct, move in enumerate(moves):
            pos = copy.deepcopy(pos1)
            com = copy.copy(positions_common)
            com.future_ojama = copy.deepcopy(positions_common.future_ojama)
            depth -= 1
            #盤面をコピー
            #盤面情報をコピー
            #moveの手を1手進める
            #進めた盤面での評価値を計算して木構造の現在ノードのscoreに格納する

            if depth == 1:
                pos.pre_move(move, com)
                score, tri_list = proc()
                name1 = 'n' + str(ct + 1) + '.score'
                name2 = 'n' + str(ct + 1) + '.move'
                setattr(self, name1, score)
                setattr(self, name3, move)
                tree_proc(move, depth, pos, ct)
            if depth == 0:
                score, tri_list = proc()
                name1 = 'n' + str(ct + 1) + '.n' + str(ct + 1) + '.score'
                name2 = 'n' + str(ct + 1) + '.n' + str(ct + 1) + '.move'
                setattr(self, name, score)
                setattr(self, name, move)
                setattr(self, name, tri_list)


root = Tree() #まず現在ノードから1手先ノードを生成する
root.init_3() #1手先ノードから2手先ノードを生成する
tree_proc(move, 2, pos, None)
# best_score, best_move = Tree.tree_proc(move, 2, )
print(root.n12.score)
