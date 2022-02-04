import sys
from enum import Enum
import numpy as np
import copy
import random


class Puyo(Enum):
    """
    ぷよの色を表す定数クラス。
    """
    EMPTY = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    PURPLE = 4
    YELLOW = 5
    OJAMA = 6

    @staticmethod
    def to_puyo(character):
        """
        ぷよを表す文字(r|g|b|p|y|o)をPuyoインスタンスに変換する。
        Parameters
        ----------
        character : str
            ぷよを表す文字。
        """
        return Puyo("ergbpyo".find(character))

    def to_str(self):
        """
        Puyoインスタンスをぷよを表す文字(r|g|b|p|y|o)に変換する。
        """
        return "ergbpyo"[int(self.value)]


class Tumo:
    """
    操作対象となっている、上から落ちてくる、ぷよが2つくっついたもの。
    Attributes
    ----------
    pivot : Puyo
        軸ぷよ
    child : Puyo
        子ぷよ
    """

    def __init__(self, c0, c1):
        self.pivot = c0
        self.child = c1


class Rule:
    """
    対戦ルールを表すためのクラス。
    Attributes
    ----------
    falltime : int
        下ボタンを押しっぱなしにしたとき、何フレームで1マス落下するか。
    chaintime : int
        1連鎖につき何フレーム硬直するか。
    settime : int
        ツモ設置時に何フレーム硬直するか。
    nexttime : int
        ツモ設置硬直後、または連鎖終了後、またはお邪魔ぷよが振り終わった後、ネクストが操作可能になるまでに何フレーム硬直するか。
    autodroptime : int
        何も操作しなかったとき、何フレームで1マス落下するか。
    """

    def __init__(self):
        self.chain_time = 60
        self.next_time = 7
        self.set_time = 15
        self.fall_time = 2
        self.autodrop_time = 50


class Move:
    """
    指し手。ツモを盤面に配置するための情報。
    Attributes
    ----------
    pivot_sq : tuple(int, int)
        軸ぷよの座標。
    child_sq : tuple(int, int)
        子ぷよの座標。
    is_tigiri : bool
        この着手がちぎりかどうか。軸ぷよのy座標 != 子ぷよのy座標のときはちぎりである。
    """

    def __init__(self, pivot_sq, child_sq, is_tigiri=False):
        self.pivot_sq = pivot_sq
        self.child_sq = child_sq
        self.is_tigiri = is_tigiri

    def to_upi(self):
        """
        指し手をupi文字列に変換する。
        """
        s0 = str(self.pivot_sq[0] + 1)
        s1 = 'abcdefghijklm'[self.pivot_sq[1]]
        s2 = str(self.child_sq[0] + 1)
        s3 = 'abcdefghijklm'[self.child_sq[1]]
        return s0 + s1 + s2 + s3

    @staticmethod
    def none():
        return Move((0, 0), (0, 0))


class Tree:
    """
    関連性行列による定形配置法に木構造を利用した処理
    1/24現在はdepth = 2にしか対応していない．Tree()の宣言部分をdepthによって動的に処理出来たら多分いける
    ----------関係-------------
    tree_proc(self, pos1, positions_common, depth):
        tree_search_impl(self, move, depth, pos1, root_count, positions_common)
            plt_proc(self, pos1)
        tree_score_calc(self)
    """

    def __init__(self):
        """
            move: このノードの着手
            total_score:未使用
            ave_score:未使用
            score:合致度
            tri_list:このノードの状態行列(1/24 未確認)
            nX:子ノードのインスタンス
        """

        self.move = Move.none()
        self.total_score = -1
        self.ave_score = -1
        self.score = -1
        self.tri_list = np.full((24, 24), -1)
        self.tier = 0
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

    def init_child_node(self):
        """
        rootの子ノードの宣言
        """
        for num in range(1, 23):
            name = 'n' + str(num)
            setattr(self, name, Tree())
            ob = getattr(self, name)
            ob.move = Move.none()
            ob.score = -1
            ob.total_score = -1
            ob.ave_score = -1
            ob.tri_list = np.full((24, 24), -1)
            ob.init_grand_chile_node()

    def init_grand_chile_node(self):
        """
        rootの孫ノードの宣言
        """
        for num in range(1, 23):
            name = 'n' + str(num)
            setattr(self, name, Tree())
            ob = getattr(self, name)
            ob.move = Move.none()
            ob.score = -1
            ob.total_score = -1
            ob.ave_score = -1
            ob.tri_list = np.full((24, 24), -1)

    def tree_search_impl(self, move, depth, pos1, root_count, positions_common):
        """
        tree_procの内部関数
        盤面を受け取って，現在から見て2手先までの全ての局面の状態行列を作成し，評価値を計算する．
        setattr(object, name, value),getattr(object, name)は動的にインスタンスを管理できる暗黒魔術
        """
        if pos1.field.is_death():
            return -999999, Move.none()
        moves = generate_moves(pos1, positions_common.tumo_pool)

        depth -= 1
        for ct, move in enumerate(moves):
            if depth == 0:
                pos = copy.deepcopy(pos1)
                com = copy.copy(positions_common)
                com.future_ojama = copy.deepcopy(positions_common.future_ojama)
                pos.pre_move(move, com)
                score, tri_list = self.plt_proc(pos)
                puyo1 = pos.field.get_puyo(move.pivot_sq[0], move.pivot_sq[1])
                puyo2 = pos.field.get_puyo(move.child_sq[0], move.child_sq[1])
                searching_pos = np.zeros((Field.X_MAX, Field.Y_MAX), dtype=bool)
                count1 = pos.field.count_connection(puyo1, move.pivot_sq[0], move.pivot_sq[1], searching_pos)
                count2 = pos.field.count_connection(puyo2, move.child_sq[0], move.child_sq[1], searching_pos)
                if count1 < 4 and count2 < 4:
                    score *= (count1 + count2) / 2
                if pos.field.floors_bounds_bool():
                    score = -9999
                if root_count != None:
                    pr_name = 'n' + str(root_count + 1)
                    myname = 'n' + str(ct + 1)
                    pr_object = getattr(self, pr_name)
                    my_object = getattr(pr_object, myname)
                    pr_score = self.score
                else:
                    my_object = getattr(self, 'n' + str(ct + 1))

                setattr(my_object, "score", score)
                setattr(my_object, "move", move)
                setattr(my_object, "tri_list", tri_list)
            if depth == 1:
                pos = copy.deepcopy(pos1)
                com = copy.copy(positions_common)
                com.future_ojama = copy.deepcopy(positions_common.future_ojama)
                pos.pre_move(move, com)
                score, tri_list = self.plt_proc(pos)
                ob1 = getattr(self, 'n' + str(ct + 1))
                setattr(ob1, "score", score)
                setattr(ob1, "move", move)
                setattr(ob1, "tri_list", tri_list)
                self.tree_search_impl(move, depth, pos, ct, positions_common)

    def tree_proc(self, pos1, positions_common, depth):
        """
        状態行列を計算する諸処理
        """
        self.tree_search_impl(None, depth, pos1, None, positions_common)
        return self.tree_score_calc()

    def tree_score_calc(self):
        """
        木構造の中から最も評価値の高いノードの1手目の評価値と着手を計算する
        計算の方法は様々考えられるが，1/24現在は葉ノードが最高となる親ノードを計算している
        """
        best_score = -9999
        best_move = Move.none()
        for ch in range(1, 23):
            for gch in range(1, 23):
                g2_object = getattr(self, "n" + str(ch))
                g1_object = getattr(g2_object, "n" + str(gch))
                move = getattr(g2_object, "move")
                score = getattr(g2_object, "score")
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

    def plt_proc(self, pos1):
        """
        盤面を受け取って，状態行列と評価値を計算する
        """

        tri_list = Field.make_tri_list(pos1.field)
        asc = Asc()
        best_score = -9999
        tier_score_list = [-1] * 8
        for lp in range(len(Data.arr_temp)):
            tem_list = asc.asc_proc(lp)
            tier_score_list[lp - 1] = Field.match_score(pos1.field, tem_list, tri_list) / Field.max_score_proc(pos1.field)

        for lp in range(len(tier_score_list)):
            if best_score < tier_score_list[lp]:
                best_score = tier_score_list[lp]
                Data.current_tier = lp + 1
                self.tier = lp + 1

        return best_score, tri_list


class Data:
    """
    関連生行列の各種データを保存しておくクラス
    Tierに関するプレイヤーデータもここに保存
    Tierの階層数をnとすると，
        arr_temp:n * X dim2
            ぷよの消え方を1~5までの数字，6はない，7以降は定形の隣接部分の空きマスにそれぞれ割り振っており，定形によって異なる．0は空きマス
        arr_temp_bias:n * X dim2
            arr_tempの要素ごとの点数を保存している．
        taboo_set:n * Y * 2 dim3
            Yはタブーの配色関係の数で，定形によって異なる．
        asc_mtx:10 * 24 * 24 dim3
            10はTierの大きさによって調整．状態行列なので4 * 6の24 * 24サイズ
        asc_mtx_score:10 * 24 * 24 dim3
            asc_mtxにスコアを付与したもの
        edge_bias:
            盤面の端から優先的に配置するようにする重み．一般に端から組んでいく方がスムーズに連鎖を組みやすいとされている
    """
    def __init__(self):
        pass

    x_bound = 6
    y_bound = 4
    size_def = x_bound * y_bound
    list_temp = np.full((y_bound, x_bound), -1)  # 定形を4*6で表したリスト

    arr_temp = [[2, 2, 3, 4, 4, 7, 1, 1, 2, 3, 3, 4, 1, 2, 3, 8, 4, 9, 10, 11, 12, 0, 13, 0],
                [2, 2, 3, 4, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 7, 4, 8, 9, 10, 11, 0, 12, 0],
                [2, 2, 3, 7, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 8, 4, 4, 9, 10, 11, 0, 12, 13],
                [2, 2, 3, 4, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 7, 4, 4, 8, 9, 10, 0, 11, 12],
                [2, 2, 3, 7, 4, 4, 1, 1, 2, 3, 3, 4, 1, 2, 3, 8, 4, 9, 10, 11, 12, 0, 13, 0],
                [2, 2, 3, 7, 8, 9, 1, 1, 2, 3, 3, 4, 1, 2, 3, 10, 11, 4, 12, 13, 14, 4, 4, 15],
                [2, 2, 3, 7, 8, 4, 1, 1, 2, 3, 3, 4, 1, 2, 3, 4, 4, 9, 10, 11, 12, 13, 14, 0],
                [2, 2, 7, 4, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 8, 4, 9, 10, 11, 12, 0, 13, 0]]  # 定形を1*24で表したリスト
    arr_temp_bias = [[500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100, 100],
                     [500, 400, 300, 200, 1000, -999999, 100, 100, 100, 100, 100, 100, 100]]
    taboo_set = [[[1, 2], [1, 10], [2, 3], [2, 11], [3, 4], [3, 7], [3, 12], [4, 7], [4, 8], [4, 13]],
                 [[1, 2], [1, 9], [2, 3], [2, 10], [3, 4], [3, 7], [3, 8], [3, 11], [4, 12]],
                 [[1, 2], [1, 9], [2, 3], [2, 10], [3, 4], [3, 7], [3, 8], [3, 11], [4, 12]],
                 [[1, 2], [1, 8], [2, 3], [2, 9], [3, 4], [3, 7], [3, 10], [4, 11], [4, 12]],
                 [[1, 2], [1, 8], [2, 3], [2, 11], [3, 4], [3, 7], [3, 8], [3, 12], [4, 7], [4, 9], [4, 11], [4, 12],
                  [4, 13]],
                 [[1, 2], [1, 12], [2, 13], [2, 10], [3, 4], [3, 7], [3, 8], [3, 10], [3, 11], [3, 14], [4, 9], [4, 11],
                  [4, 12], [4, 13], [4, 14], [4, 15]],
                 [[1, 2], [1, 10], [2, 3], [2, 11], [3, 4], [3, 7], [3, 8], [3, 12], [4, 9], [4, 13], [4, 14]],
                 [[1, 2], [1, 10], [2, 3], [2, 7], [2, 11], [3, 4], [3, 8], [3, 9], [3, 12], [4, 7], [4, 9], [4, 13]]]
    edge_bias = [2.5, 2.1, 1.0, 1.0, 1.1, 1.5]
    asc_mtx = np.full((10, 24, 24), -1)
    asc_mtx_score = np.full((10, 24, 24), -1.0)
    current_tier = 1

    def current(tier):
        """
        tierを受け取り，Data内の各種データを返却する
        :param tier:
        :return:
        """
        arr = Data.arr_temp[tier]
        bias = Data.arr_temp_bias[tier]
        taboo = Data.taboo_set[tier]
        asc = Data.asc_mtx[tier]
        asc_score = Data.asc_mtx_score[tier]
        return arr, bias, taboo, asc, asc_score

    def tier_print(Data):
        tmp_list = np.full((4, 6), -1)
        for lp in range(len(Data.arr_temp)):
            for row in range(len(Data.arr_temp[0])):
                pass


class Asc:
    """
    テンプレート行列を返却するためのクラス
    Dataクラスからデータをロードし，テンプレート行列を作成する
    """

    def __init__(self):
        self.list_temp = np.full((Data.y_bound, Data.x_bound), -1)  # 定形を4*6で表したリスト

    def asc_proc(self, tier):
        """
        テンプレート行列を作るメインメソッド
        :return: テンプレート行列
        """

        return Data.asc_mtx_score[tier]

    def make_asc_mtx(self, arr, taboo, asc):
        """
        定形の状態行列を返却する
        """
        self.list_temp = np.array(self.arr_to_list(arr))
        for row in range(self.list_temp.shape[0]):
            for col in range(self.list_temp.shape[1]):
                color = self.list_temp[row][col]
                for in_row in range(self.list_temp.shape[0]):
                    for in_col in range(self.list_temp.shape[1]):
                        tmp_color = self.list_temp[in_row][in_col]
                        y_ = row * 6 + col
                        x_ = in_row * 6 + in_col
                        if self.bool_taboo_check(taboo, row, col, in_row, in_col):
                            asc[y_][x_] = -1
                        elif color == 0 or tmp_color == 0:
                            asc[y_][x_] = 0
                        elif color != tmp_color:
                            if self.bool_next_to(row, col, in_row, in_col):
                                asc[y_][x_] = -1
                            else:
                                asc[y_][x_] = 0
                        elif color == tmp_color:
                            asc[y_][x_] = 1
                        else:
                            print("ex I am", row, col, in_row, in_col)
        return asc

    def asc_mtx_score(self, asc, asc_score, tier):
        """
        重みを付与して，テンプレート行列を返却する
        :param asc:
        :param asc_score:
        :return:
        """
        arr_temp = Data.arr_temp[tier]
        arr_bias = Data.arr_temp_bias[tier]
        for row in range(len(arr_temp)):
            for col in range(len(arr_temp)):
                try:
                    if asc[row][col] == 0:
                        asc_score[row][col] = 0
                    elif asc[row][col] == 1:
                        asc_score[row][col] = (arr_bias[arr_temp[row] - 1] + arr_bias[arr_temp[col] - 1]) / 2
                    elif asc[row][col] == -1:
                        asc_score[row][col] = (arr_bias[arr_temp[row] - 1] + arr_bias[arr_temp[col] - 1]) / 2 * -1
                    else:
                        print("ex I am", row, col, asc[row][col])
                except IndexError:
                    try:
                        arr_bias[arr_temp[row]]
                    except IndexError:
                        print("arr_bias index is strange!!\n")
                        print(arr_temp[row]," ",arr_temp[col], "are assigning")
        return asc_score

    def asc_ready(self):
        for lp in range(len(Data.arr_temp)):
            tier = lp
            value = Data.current(tier)
            asc_ = self.make_asc_mtx(value[0], value[2], value[3])
            Data.asc_mtx[tier] = asc_
            asc_score = self.asc_mtx_score(asc_, value[4], tier)
            asc_score = self.add_score_edge(asc_score)
            Data.asc_mtx_score[tier] = asc_score
        pass

    def arr_to_list(self, arr):
        """
        1次元の定石リストを4*6の二次元リストに変換して返却
        """
        temp_list = np.full((Data.y_bound, Data.x_bound), -1)
        for lp in range(len(arr)):
            row = lp // 6
            col = lp % 6
            temp_list[row][col] = arr[lp]
        return temp_list

    def bool_next_to(self, row, col, in_row, in_col):
        """
        (row, col)と(in_row, in_col)のマスが左右上下に隣接していたらTrue,Falseを返す
        """
        if row == in_row - 1 and col == in_col:
            return True
        elif row == in_row + 1 and col == in_col:
            return True
        elif row == in_row and col == in_col - 1:
            return True
        elif row == in_row and col == in_col + 1:
            return True
        else:
            return False

    def bool_taboo_check(self, taboo, row, col, in_row, in_col):
        """
        参照マスと走査マスがXの組み合わせならFalseをTrueを返却する
        """
        value1 = self.list_temp[row][col]
        value2 = self.list_temp[in_row][in_col]
        set1 = [value1, value2]
        set2 = [value2, value1]
        for lp in range(len(taboo)):
            if taboo[lp] == set1 or taboo[lp] == set2:
                return True
            else:
                return False

    def add_score_edge(self, asc_score):
        for row in range(Data.size_def):
            for col in range(Data.size_def):
                asc_score[row][col] = asc_score[row][col] * Data.edge_bias[row % 6]

        return asc_score


class Field:
    """
    盤面。ツモを配置する空間。
    Attributes
    ----------
    field : np.ndarray(Puyo)
        6行13列のPuyo配列。
    """
    X_MAX = 6
    Y_MAX = 13

    def __init__(self):
        self.field = np.full((self.X_MAX, self.Y_MAX), Puyo.EMPTY)

    def init_from_pfen(self, pfen):
        """
        pfen文字列からぷよ配列を初期化する。
        Parameters
        ----------
        pfen : str
            pfen文字列の盤面部分のみ
        """
        x = 0
        y = 0
        self.__init__()
        for p in pfen:
            if p == "/":
                x += 1
                y = 0
            else:
                self.set_puyo(x, y, Puyo.to_puyo(p))
                y += 1

    def set_puyo(self, x, y, col):
        self.field[x, y] = col

    def get_puyo(self, x, y):
        return self.field[x, y]

    @staticmethod
    def is_in_field(x, y):
        """
        座標がフィールドの見えている範囲内にあるかどうかを判定する。
        """
        return x >= 0 and x < Field.X_MAX and y >= 0 and y < Field.Y_MAX - 1

    def pretty_print(self):
        """
        Fieldインスタンスを色付きで見やすく標準出力に表示する。
        """
        color = ('\033[30m', '\033[31m', '\033[32m', '\033[34m', '\033[35m', '\033[33m', '\033[37m')
        END = '\033[0m'
        pretty_string = self.pretty()
        for p in pretty_string:
            id = "ergbpyo".find(p)
            if id >= 0:
                print(color[id] + p + END, end='')
            else:
                print(p, end='')
        print('')

    def pretty(self):
        """
        Fieldインスタンスを見やすい文字列に変換する。
        """
        result = ''
        for y in reversed(range(self.Y_MAX)):
            for x in range(self.X_MAX):
                result += self.get_puyo(x, y).to_str()
            result += '\r\n'
            if y == 12:
                result += '------\r\n'
        return result[:-2]

    def pretty_num(self):
        """
        Filedインスタンスを数字列に変換する．
        """
        result = ''
        for y in reversed(range(self.Y_MAX)):
            for x in range(self.X_MAX):
                result += str(self.get_puyo(x, y).value)
            result += '\r\n'
            if y == 12:
                result += '------\r\n'
        print(result)
        return result

    def make_numlist(self):  # add
        """
        fieldを数字リストとして返す．
        """
        arr = np.full((self.Y_MAX, self.X_MAX), Puyo.EMPTY)
        for y in reversed(range(self.Y_MAX)):
            for x in range(self.X_MAX):
                arr[self.Y_MAX - 1 - y, x] = self.get_puyo(y, x).value
        return arr

    def is_empty(self):
        """
        フィールドがすべて空かを判定する。
        """
        return np.any(self.field) == Puyo.EMPTY

    def count_connection(self, puyo, x, y, searched):
        """
        指定された座標にあるぷよの連結数を計算する。
        """
        if not self.is_in_field(x, y) or searched[x, y] or self.get_puyo(x, y) != puyo:
            return 0
        searched[x, y] = True
        return (self.count_connection(puyo, x - 1, y, searched) +
                self.count_connection(puyo, x + 1, y, searched) +
                self.count_connection(puyo, x, y - 1, searched) +
                self.count_connection(puyo, x, y + 1, searched) + 1)

    def calc_delete_puyo(self, chain_num):
        """
        4つ以上つながっている場所と、この連鎖でのスコアを計算する。
        Parameters
        ----------
        chain_num : int
            現盤面での連鎖数。初めての連鎖なら0。
        Returns
        -------
            score : int
                この連鎖でのスコア。
            delete_pos : np.ndarray(bool)
                消える場所がTrueになっているarray。
        """
        CHAIN_BONUS = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)
        CONNECT_BONUS = (0, 2, 3, 4, 5, 6, 7, 10)
        COLOR_BONUS = (0, 3, 6, 12, 24)
        searched_pos = np.zeros((self.X_MAX, self.Y_MAX), dtype=bool)
        delete_pos = np.zeros((self.X_MAX, self.Y_MAX), dtype=bool)
        colors = {}
        score = 0
        for x in range(self.X_MAX):
            for y in range(self.Y_MAX - 1):
                puyo = self.get_puyo(x, y)
                if puyo == Puyo.EMPTY:
                    break
                elif puyo != Puyo.OJAMA and not searched_pos[x, y]:
                    searching_pos = np.zeros((self.X_MAX, self.Y_MAX), dtype=bool)
                    count = self.count_connection(puyo, x, y, searching_pos)
                    searched_pos |= searching_pos
                    if count >= 4:
                        delete_pos |= searching_pos
                        colors[puyo] = 1
                        score += CONNECT_BONUS[min(count, 11) - 4]
        if len(colors) > 0:
            score += CHAIN_BONUS[chain_num] + COLOR_BONUS[len(colors) - 1]
            score = np.count_nonzero(delete_pos) * max(score, 1) * 10
        return score, delete_pos


    def delete_puyo(self, delete_pos):
        """
        引数で与えられた場所を空にする。消えるぷよの上下左右1マス以内にお邪魔ぷよがあれば消す。

        Parameters
        ----------
        delete_pos : np.ndarray(bool)
            消える場所がTrueになっているarray。
        """
        pos = np.where(delete_pos)
        for x, y in zip(pos[0], pos[1]):
            self.set_puyo(x, y, Puyo.EMPTY)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if self.is_in_field(x + dx, y + dy) and self.get_puyo(x + dx, y + dy) == Puyo.OJAMA:
                    self.set_puyo(x + dx, y + dy, Puyo.EMPTY)

    def slide(self):
        """
        ぷよを消した後、落下するぷよがあれば着地するまで落下させる。
        """
        for x in range(self.X_MAX):
            for y in range(self.Y_MAX):
                if self.get_puyo(x, y) == Puyo.EMPTY:
                    top_y = y
                    while y < self.Y_MAX and self.get_puyo(x, y) == Puyo.EMPTY:
                        y += 1
                    if y >= self.Y_MAX:
                        break
                    self.set_puyo(x, top_y, self.get_puyo(x, y))
                    self.set_puyo(x, y, Puyo.EMPTY)

    def chain(self):
        """
        連鎖を最後まで行う。
        Returns
        -------
        chain_num : int
            連鎖数。
        score_num : int
            この連鎖のスコア。
        """
        chain_num = 0
        score_sum = 0
        while True:
            score, delete_pos = self.calc_delete_puyo(chain_num)  # ここが全てzeroで通っている
            if score == 0:
                # print("zero")
                break
            else:
                self.delete_puyo(delete_pos)
                self.slide()
                chain_num += 1
                score_sum += score
        return (chain_num, score_sum)

    def is_death(self):
        """
        死んでいるフィールドかを判定する。
        """
        return self.get_puyo(2, 11) != Puyo.EMPTY

    def floors(self):
        """
        床座標を返す。
        Returns
        ------
        floor_y : list(int)
            列ごとの床座標。何もないフィールドなら、[0, 0, 0, 0, 0, 0]。
        """
        floor_y = [self.Y_MAX] * 6
        for x in range(self.X_MAX):
            for y in range(self.Y_MAX):
                if self.get_puyo(x, y) == Puyo.EMPTY:
                    floor_y[x] = y
                    break
        return floor_y

    def floors_bounds_bool(self):
        """
        配置したぷよの高さが6以上であれば真，以外は偽
        :return: Bool
        """
        if np.amax(self.floors()) > 5:
            return True
        else:
            return False

    """
    Attributes
    ----------
    match_score 
        合致度を計算する．0 <= return <= 1
    max_score_proc
        合致scoreの最大を計算する．
    make_tri_list
        与えられた盤面(y,x)から，(y*x)*(y*x)の状態行列を作る．

    """
    X_match = 6
    Y_match = 4
    size_def = X_match * Y_match
    temp_arr = np.full((size_def, size_def), -1)

    def match_score(self, temp_arr, tri_list):
        """
        :param temp_arr: テンプレート行列
        :param tri_list: 状態行列
        :return:その着手の点数
        """
        score = 0
        for i in range(self.X_match * self.Y_match):
            for j in range(self.X_match * self.Y_match):
                score += temp_arr[i][j] * tri_list[i][j]
                if temp_arr[i][j] * tri_list[i][j] < 0:
                    score -= 1000000
        return score

    def max_score_proc(self):
        """
        スコアを割合表示するために評価値の最大を計算する
        :return: 最大値
        """
        max_score = 0
        for lp1 in range(self.size_def):
            for lp2 in range(self.size_def):
                if Data.asc_mtx_score[Data.current_tier][lp1][lp2] == 0 or Data.asc_mtx_score[Data.current_tier][lp1][lp2] == 0:
                    max_score += 0
                    continue
                else:
                    max_score += (abs(Data.asc_mtx_score[Data.current_tier][lp1][lp2]) + abs(Data.asc_mtx_score[Data.current_tier][lp1][lp2])) / 2

        return max_score

    def make_tri_list(self):
        """
        Parameters
        tri_list
        field
        引数に1pの盤面情報も必要(仮field)
        Attributes
        ---------
        color そのマスが何色かを示す(0,1,2,3,4)
        i,j そのマスに対して状態行列を作るために，どこから走査するかを記録する変数
        """
        tri_list = np.full((self.Y_match * self.X_match, self.Y_match * self.X_match), 0)
        trans_field = self.field.T.copy()
        for y in range(self.Y_match):
            for x in range(self.X_match):
                """
                [row][col]のマスから見たそれ以外のマスに対して，同色なら+1，異色なら-1，空きマスなら0を入れる
                """
                color = trans_field[y][x]
                for i in range(y, self.Y_match):
                    for j in range(x, self.X_match):
                        if color == Puyo.EMPTY or trans_field[i][j] == Puyo.EMPTY:
                            tri_list[y * 6 + x][i * 6 + j] = 0
                            continue
                        elif trans_field[i][j] == color:
                            tri_list[y * 6 + x][i * 6 + j] = 1
                            continue
                        elif trans_field[i][j] != color:
                            tri_list[y * 6 + x][i * 6 + j] = -1
                            continue
                        else:
                            pass
        return tri_list


class Position:
    """
    Fieldインスタンス、ツモ、スコア、お邪魔を管理するクラス。
    Attributes
    ----------
    field : Field
        Fieldインスタンス。
    tumo_index : int
        ツモ番号。配ツモ自体は外部から引数で与えられる。
    ojama_index : int
        お邪魔ぷよ乱数。
    fall_bonus : int
        落下ボーナス。
    all_clear_flag : bool
        全消しフラグ。
    """

    def __init__(self):
        self.field = Field()
        self.tumo_index = 0
        self.ojama_index = 0
        self.fall_bonus = 0
        self.all_clear_flag = False
        self.rule = Rule()

    def fall_ojama(self, positions_common):
        """
        確定予告ぷよをおじゃまぷよとして盤面に配置する。
        """
        floors = self.field.floors()
        ojama = min(30, positions_common.future_ojama.fixed_ojama)
        # 6個以上降る場合は、まず6の倍数個降らせる。
        while ojama >= Field.X_MAX:
            for x in range(Field.X_MAX):
                if floors[x] < Field.Y_MAX:
                    self.field.set_puyo(x, floors[x], Puyo.OJAMA)
                    floors[x] += 1
                ojama -= 1
                self.ojama_index = self.ojama_index + 1 % 128
        # ここまで来ると確定予告ぷよは6個未満になっているはず。
        assert ojama < 6
        if ojama > 0:
            # サーバと同じロジックでお邪魔ぷよが降る場所を決める。
            v = list(range(6))
            for x in range(Field.X_MAX):
                t = positions_common.tumo_pool[self.ojama_index]
                r = (t.pivot.value + t.child.value) % 6
                v[x], v[r] = v[r], v[x]
                self.ojama_index = self.ojama_index + 1 % 128
            for x in range(ojama):
                if floors[v[x]] < Field.Y_MAX:
                    self.field.set_puyo(v[x], floors[v[x]], Puyo.OJAMA)

    def pre_move(self, move, positions_common):  # Add
        """
        -追加メソッド-
        着手はするが，4連結以上が生まれても消さない
        Returns
        -----
        chain_num:
            その座標の連結数
        score:
            評価値
        """
        tumo = positions_common.tumo_pool[self.tumo_index]
        rule = positions_common.rule
        future_ojama = positions_common.future_ojama
        self.tumo_index = (self.tumo_index + 1) % 128
        p = move.pivot_sq
        c = move.child_sq
        self.field.set_puyo(p[0], p[1], tumo.pivot)
        self.field.set_puyo(c[0], c[1], tumo.child)
        chain_num, score = self.field.chain()
        return (chain_num, score)

    def do_move(self, move, positions_common):
        """
        指し手に応じて盤面を次の状態に進める。着手→連鎖→お邪魔ぷよ落下までを行う。
        Parameters
        ----------
        move : Move
            指し手。
        positions_common : PositionsCommonInfo
            配ツモ、ルール、予告ぷよ。
        """
        tumo = positions_common.tumo_pool[self.tumo_index]
        rule = positions_common.rule
        future_ojama = positions_common.future_ojama
        self.tumo_index = (self.tumo_index + 1) % 128
        p = move.pivot_sq
        c = move.child_sq
        self.field.set_puyo(p[0], p[1], tumo.pivot)
        self.field.set_puyo(c[0], c[1], tumo.child)
        chain_num, score = self.field.chain()
        if chain_num > 0:
            if self.all_clear_flag:
                score += 70 * 30
                self.all_clear_flag = False
            if self.field.is_empty():
                self.all_clear_flag = True
            score += self.fall_bonus
            ojama = int(score / 70)
            self.fall_bonus = score % 70
            # おじゃまぷよ相殺。相殺しきれば相手の未確定予告ぷよとして返す。
            if future_ojama.fixed_ojama > 0:
                future_ojama.fixed_ojama -= ojama
                if future_ojama.fixed_ojama < 0:
                    future_ojama.unfixed_ojama += future_ojama.fixed_ojama
                    future_ojama.fixed_ojama = 0
            else:
                future_ojama.unfixed_ojama -= ojama

        drop_frame = max(12 - p[1], 12 - c[1]) * rule.fall_time
        frame = (drop_frame + max(abs(2 - p[0]), abs(2 - c[0]))
                 + rule.set_time * 2 if move.is_tigiri else rule.set_time
                                                            + rule.chain_time * chain_num
                                                            + rule.next_time)
        if future_ojama.unfixed_ojama > 0:
            future_ojama.time_until_fall_ojama -= frame
            if future_ojama.time_until_fall_ojama <= 0:
                future_ojama.fixed_ojama += future_ojama.unfixed_ojama
                future_ojama.unfixed_ojama = 0
                future_ojama.time_until_fall_ojama = frame
        if future_ojama.fixed_ojama > 0:
            self.fall_ojama(positions_common)


class FutureOjama:
    """
    予告ぷよ。
    Attributes
    ----------
    fixed_ojama : int
        確定予告ぷよ。着手を行ったときに降ることが確定している。
    unfixed_ojama : int
        未確定予告ぷよ。着手を行っても降らない。
    time_until_fall_ojama : int
        未確定予告ぷよが確定予告ぷよになるまでのフレーム数。
    """

    def __init__(self):
        self.fixed_ojama = 0
        self.unfixed_ojama = 0
        self.time_until_fall_ojama = 0


class PositionsCommonInfo:
    """
    1Pの局面と2Pの局面で共通しているデータ。
    Attributes
    ----------
    tumo_pool : list(Tumo)
        配ツモ。
    rule : Rule
        ルール。
    future_ojama : FutureOjama
        予告ぷよ。
    """

    def __init__(self):
        self.tumo_pool = [Tumo(Puyo(i % 4 + 1), Puyo((i + 1) % 4 + 1)) for i in range(128)]  # 適当に初期化
        self.rule = Rule()
        self.future_ojama = FutureOjama()


def generate_moves(pos, tumo_pool):
    """
    この局面で着手可能な指し手のリストを生成する。
    Parameters
    ----------
    pos : Position
        局面。
    tumo_pool : list(Tumo)
        配ツモ。

    Returns
    -------
    moves : list(Move)
        着手可能な指し手のリスト。
    """
    floors = pos.field.floors()
    start_x, end_x = get_move_range(floors)
    moves = []
    tumo = tumo_pool[pos.tumo_index]
    for x in range(start_x, end_x):
        y = floors[x]
        y_side = floors[x + 1]
        dest = (x, y)
        dest_up = (x, y + 1)
        dest_side = (x + 1, y_side)
        is_tigiri = (y != y_side)
        moves.append(Move(dest, dest_up, False))
        moves.append(Move(dest, dest_side, is_tigiri))
        if tumo.pivot != tumo.child:
            moves.append(Move(dest_up, dest, False))
            moves.append(Move(dest_side, dest, is_tigiri))
    dest = (end_x, floors[end_x])
    dest_up = (end_x, floors[end_x] + 1)
    moves.append(Move(dest, dest_up, False))
    if tumo.pivot != tumo.child:
        moves.append(Move(dest_up, dest, False))
    return moves


def get_move_range(floors):
    """
    何列目から何列目までが着手可能なのかを返す。
    Parameters
    ----------
    floors : list(int)
        床座標。

    Returns
    -------
    left : int
        着手可能なx座標の最小値。
    right : int
        着手可能なx座標の最大値。
    """
    left = 0
    right = 5
    for x in reversed(range(2)):
        if floors[x] >= 12:
            left = x + 1
            break
    for x in range(3, Field.X_MAX):
        if floors[x] >= 12:
            right = x - 1
            break
    return (left, right)


def search(pos1, pos2, positions_common, depth):
    """
    この局面での最善手を探索する。
    Parameters
    ----------
    pos1 : Position
        1Pの局面。
    pos2 : Position
        2Pの局面。
    positions_common : PositionsCommonInfo
        1Pと2Pで共通のデータ。
    depth : int
        探索深さ。

    Returns
    -------
    move : Move
        探索した結果の指し手。
    """

    if pos1.field.is_death():
        return Move.none()
    moves = generate_moves(pos1, positions_common.tumo_pool)
    # score, move = search_impl(pos1, pos2, positions_common, depth)
    score, move = tree_root(pos1, positions_common, depth)
    if move.to_upi() == Move.none().to_upi():
        return moves[0]
    pos1.do_move(move, positions_common)
    Field.pretty_print(pos1.field)  # 設置後の盤面を表す

    if move == Move.none():
        return False
    # if pos1.field.floors_bounds_bool():  # 範囲外に置くと強制終了する
    #     sys.exit()
    return move


def tree_root(pos1, positions_common, depth):
    root = Tree()
    root.init_child_node()
    best_score, best_move = Tree.tree_proc(root, pos1, positions_common, depth)
    print("↓Tier", root.tier)

    return best_score, best_move


class UpiPlayer:
    def __init__(self):
        self.common_info = PositionsCommonInfo()
        self.positions = [Position(), Position()]

    def upi(self):
        engine_name = "Hanna"
        version = "1.0"
        author = "Takato Kobayashi"
        print("id name", engine_name + version)
        print("id author", author)
        print("upiok")

    def tumo(self, tumos):
        self.common_info.tumo_pool = [Tumo(Puyo.to_puyo(t[0]), Puyo.to_puyo(t[1])) for t in tumos]

    def rule(self, rules):
        for i in range(0, len(rules), 2):
            if rules[i] == "falltime":
                self.common_info.rule.fall_time = int(rules[i + 1])
            elif rules[i] == "chaintime":
                self.common_info.rule.chain_time = int(rules[i + 1])
            elif rules[i] == "settime":
                self.common_info.rule.set_time = int(rules[i + 1])
            elif rules[i] == "nexttime":
                self.common_info.rule.next_time = int(rules[i + 1])
            elif rules[i] == "autodroptime":
                self.common_info.rule.autodrop_time = int(rules[i + 1])

    def isready(self):
        asc = Asc()
        asc.asc_ready()
        print("readyok")

    def position(self, pfen):
        for i in range(2):
            self.positions[i].field.init_from_pfen(pfen[i * 2])
            self.positions[i].tumo_index = int(pfen[i * 2 + 1])
        self.common_info.future_ojama.fixed_ojama = int(pfen[4])
        self.common_info.future_ojama.unfixed_ojama = int(pfen[5])
        self.common_info.future_ojama.time_until_fall_ojama = int(pfen[6])

    def go(self):
        move = search(self.positions[0], self.positions[1], self.common_info, 1)

        print('bestmove', move.to_upi())

    def gameover(self):
        # 特に何もしない
        sys.exit()
        pass

    def tumo_edit(self, text):
        out_list = [' '] * 64
        for lp in range(0, 128, 2):
            out_list.insert(lp + 1, text[lp:lp + 2])
        out_txt = ''.join(out_list)
        return out_txt

    def tumo_read(self):
        size = 128
        tumo_tmp = [''] * 65536
        tumo = [''] * 65536
        flread = open('haipuyo.txt', 'r')
        count_inner = 0
        for line in flread:
            tumo_tmp[count_inner] = line + "\n"
            tumo[count_inner] = 'tumo' + self.tumo_edit(tumo_tmp[count_inner])
            count_inner += 1

        return tumo


if __name__ == "__main__":
    tumo_index_count = random.randint(0,65535)
    count = 0
    token = "isready"
    upi = UpiPlayer()
    tumos = upi.tumo_read()
    while token != "quit":
        # cmd = input().split(' ')
        # token = cmd[0]

        # UPIエンジンとして認識されるために必要なコマンド
        if token == "upi":
            upi.upi()

        elif token == "next":
            upi.__init__()
            cmd = tumos[tumo_index_count].split(' ')
            upi.tumo(cmd[1:])
            count = 1
            print("ツモ番号 ", tumo_index_count, "\n")
            tumo_index_count = random.randint(0, 65535)
            token = "go"

        # 今回のゲームで使うツモ128個
        # elif token == "tumo":
        #     upi.tumo(cmd[1:])
        #
        #     # ルール
        # elif token == "rule":
        #     upi.rule(cmd[1:])

        # 時間のかかる前処理はここで。
        elif token == "isready":
            upi.isready()
            token = "next"
            data = Data()
            # 思考開始する局面を作る。
        # elif token == "position":
        #     upi.position(cmd[1:])

        # 思考開始の合図。エンジンはこれを受信すると思考を開始。
        elif token == "go":
            count += 1
            turn = 12
            upi.go()
            if count > turn:
                Field.pretty_print(upi.positions[0].field)
                print(turn, "ターン経過..", "ツモ番号", tumo_index_count, "での連鎖構築を終了します")
                token = "next"
            if tumo_index_count > 65535:
                print("ツモ番号 ", tumo_index_count, "終了します．\n")
                exit()

            # ゲーム終了時に送ってくるコマンド
        elif token == "gameover":
            upi.gameover()

        # 有効なコマンドではない。
        # else:
        #     print("unknown command: ", cmd)
