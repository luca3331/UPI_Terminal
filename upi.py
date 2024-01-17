from enum import Enum
import numpy as np
import copy


# 思考エンジン用ライブラリのimport
import torch
import torch.nn as nn
# from pfen2kifdata import InputData
# from modeling import SimpleNet
from modeling import Net

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
            score, delete_pos = self.calc_delete_puyo(chain_num)
            if score == 0:
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
        self.field_pfen = ''  # 追加コンテンツ 20240112
    
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
        self.tumo_pool = [Tumo(Puyo(i % 4 + 1), Puyo((i + 1) % 4 + 1)) for i in range(128)] # 適当に初期化
        self.rule = Rule()
        self.future_ojama = FutureOjama()
        self.tumo_pfen = '' # 追加コンテンツ 20240112

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
    score, move = search_impl(pos1, pos2, positions_common, depth)
    if move.to_upi() == Move.none().to_upi():
        return moves[0]
    return move

def search_impl(pos1, pos2, positions_common, depth):
    """
    evaluateの返り値が最も大きくなる手を探索する。

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
    best_score : int
        最も大きかったevaluateの返り値。
    best_move : Move
        best_scoreを得ることができる指し手。
    """

    if depth == 0:
        return evaluate(pos1, positions_common), Move.none()
    if pos1.field.is_death():
        return -999999, Move.none()
    moves = generate_moves(pos1, positions_common.tumo_pool)
    best_score = -999999
    best_move = Move.none()
    for move in moves:
        pos = copy.deepcopy(pos1)
        com = copy.copy(positions_common)
        com.future_ojama = copy.deepcopy(positions_common.future_ojama)
        pos.do_move(move, com)
        score, _ = search_impl(pos, pos2, com, depth - 1)
        if score > best_score:
            best_score = score
            best_move = move
    return best_score, best_move

def evaluate(pos, positions_common):
    """
    局面を評価する。単純に、お邪魔ぷよの数で判定する。

    Parameters
    ----------
    pos : Position
        評価する局面。
    positions_common : PositionsCommonInfo
        共通データ。お邪魔ぷよを含む。
    
    Returns
    -------
    eval : int
        局面のスコア。今のところお邪魔ぷよの数。相手に降らせる数が多いほどハイスコア。
    """
    if pos.field.is_death():
        return -999999
    return -(positions_common.future_ojama.fixed_ojama + positions_common.future_ojama.unfixed_ojama)


def cnn_predict(model, pos1, pos2, positions_common, depth):
    # p1_field_pfen, p1_tumonum = pos1.field_pfen[0], int(pos1.field_pfen[1])
    input_data = InputData(pos1.field_pfen, positions_common.tumo_pfen)
    with torch.no_grad():
        field_tensor, tumonext = torch.tensor(input_data.encoded_field).float(), torch.tensor(input_data.encoded_others).float()
        x1 = torch.unsqueeze(field_tensor.view(6, 12, 6), 0)
        tumonext = torch.unsqueeze(tumonext, 0)
        outputs = model(x1, tumonext)
        top_scores, top_indices = torch.topk(outputs, k=22)
        # predict = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    move = cnnhand2upi(top_indices[0][0].item() + 1, pos1)
    upi.prev_predicts = list(top_indices[0][1:])
    return move


def cnnhand2upi(predict, pos):
    floors = pos.field.floors()
    try:
        if 1 <= predict < 7:
            pivot_x, child_x = predict, predict
            pivot_y, child_y = floors[pivot_x - 1] + 1, floors[child_x - 1] + 2
        elif 7 <= predict < 13:
            pivot_x, child_x = predict - 6, predict - 6
            pivot_y, child_y = floors[pivot_x - 1] + 2, floors[child_x - 1] + 1
        elif 13 <= predict < 18:
            pivot_x, child_x = predict - 12 , predict - 12 + 1
            pivot_y, child_y = floors[pivot_x - 1] + 1, floors[child_x - 1] + 1
        elif 18 <= predict:
            pivot_x, child_x = predict - 17 + 1, predict - 17
            pivot_y, child_y = floors[pivot_x - 1] + 1, floors[child_x - 1] + 1
        s0 = str(pivot_x)
        s1 = 'abcdefghijklm'[pivot_y - 1]
        s2 = str(child_x)
        s3 = 'abcdefghijklm'[child_y - 1]
        return s0 + s1 + s2 + s3
    except IndexError:
        return "0a0a"



class InputData:
    def __init__(self, command, tumos):
        self.p1_field_pfen, self.p1_tumonum, self.p2_field_pfen, self.p2_tumonum, self.val1, self.val2, self.val3 = command
        self.p1_tumonum, self.p2_tumonum = int(self.p1_tumonum), int(self.p2_tumonum)
        self.color_dict = {'r': 1, 'g': 2, 'b': 3, 'y': 4, 'p': 5, 'o': 6}
        self.p1_field_kif, self.p2_field_kif = self.pfen2kif(self.p1_field_pfen), self.pfen2kif(self.p2_field_pfen)
        self.tumos = tumos
        self.input_data = self.encode_data(self.p1_field_kif, int(self.p1_tumonum))
        self.encoded_field = self.input_data[:432]
        self.encoded_others = self.input_data[432:]
        pass


    def encode_data(self, field_kif, tumonum):
        _field = self.encode_field(field_kif)
        _tumo = self.encode_next(self.tumos[tumonum % len(self.tumos)])
        _next1 = self.encode_next(self.tumos[tumonum + 1 % len(self.tumos)])
        _next2 = self.encode_next(self.tumos[tumonum + 2 % len(self.tumos)])
        return *_field, *_tumo, *_next1, *_next2


    def pfen2kif(self, pfen):
        cols = pfen.split('/')
        puyo_exist = []
        for c_i, col in enumerate(cols):
            for r_i, row in enumerate(col):
                puyo_exist.append((c_i, r_i, self.color_dict[row]))

        field = [0] * 72
        for ele in puyo_exist:
            point = ele[1] + ele[0] * 12
            field[point] = ele[2]
        field[35] = 7
        field = ''.join(map(str,field))
        return field

    def encode_field(self, field):
        encoded_field = [0] * 12 * 6 * 6
        for i, cell in enumerate(field):
            color = int(cell)
            if color == 7 or color == 0:  # 窒息のばってん or 空きマス
                pass
            else:
                pos = (color - 1) * 72 + i
                encoded_field[pos] = 1
        return encoded_field

    def encode_next(self, tumo):
        encoded_tumo = [0] * 2 * 5
        for i, puyo in enumerate(tumo):
            color = int(self.color_dict[puyo])
            pos = (color - 1) * 2 + i
            encoded_tumo[pos] = 1
        return encoded_tumo



class UpiPlayer:
    def __init__(self):
        self.common_info = PositionsCommonInfo()        
        self.positions = [Position(), Position()]
        self.cnn_model = Net()
        self.prev_predicts = []

    def upi(self):
        engine_name = "cnn_test01"
        version = "1.0"
        author = "Takato kobayashi"
        print("id name", engine_name + version)
        print("id author", author)
        print("upiok")

    def tumo(self, tumos):
        self.common_info.tumo_pool = [Tumo(Puyo.to_puyo(t[0]), Puyo.to_puyo(t[1])) for t in tumos]
        self.common_info.tumo_pfen = tumos

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
        model_path = './_model/model_100_epochs.pth'
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.cnn_model.load_state_dict(model_dict)
        self.cnn_model.eval()
        print("readyok")

    def position(self, pfen):
        for i in range(2):
            self.positions[i].field.init_from_pfen(pfen[i * 2])            
            self.positions[i].tumo_index = int(pfen[i * 2 + 1])
            self.positions[i].field_pfen = pfen
        self.common_info.future_ojama.fixed_ojama = int(pfen[4])
        self.common_info.future_ojama.unfixed_ojama = int(pfen[5])
        self.common_info.future_ojama.time_until_fall_ojama = int(pfen[6])

    def go(self):
        move = cnn_predict(self.cnn_model, self.positions[0], self.positions[1], self.common_info, 2)
        print('bestmove', move)

    def change_2ndmove(self):
        move = cnnhand2upi(self.prev_predicts.pop().item() + 1, self.positions[0])
        print('bestmove', move)


    def gameover(self):
        # 特に何もしない
        pass


if __name__ == "__main__":
    token = ""
    upi = UpiPlayer()
    while token != "quit":
        cmd = input().split(' ')
        token = cmd[0]
                
        # UPIエンジンとして認識されるために必要なコマンド
        if token == "upi":
            upi.upi()

        # 今回のゲームで使うツモ128個
        elif token == "tumo":
            # upi.tumo("ry rb gy gb rg yy rb gg yg rg yb gr rr yy by gr bb rb yb bg rb gy gy gy yy yr yg yy rg rb bg br rr by gy rb bg gg yr yy rg rb by by gb yy bg bg rg yg rg yb yg yr yr rr br br gr gb rr yr rb gg gg bg gy bb yg gg gb yr yg yb bg br gy br rb gy rr rr yb gy by gb yy gr bb yy gb gr rr by yb gb yr yg rr gg yy yy gy by gr br rg gy yg gr yb by bg yr yr bb rr br yr gr gr yg bb yr rr gb yg gy".split(' '))
            upi.tumo(cmd[1:])

        # ルール
        elif token == "rule": 
            upi.rule(cmd[1:]) 

        # 時間のかかる前処理はここで。
        elif token == "isready": 
            upi.isready() 

        # 思考開始する局面を作る。
        elif token == "position": 
            upi.position(cmd[1:]) 

        # 思考開始の合図。エンジンはこれを受信すると思考を開始。
        elif token == "go": 
            upi.go() 

        # ゲーム終了時に送ってくるコマンド
        elif token == "gameover": 
            upi.gameover()

        elif token == "illegalmove":
            upi.change_2ndmove()

        # 有効なコマンドではない。
        else:
            print("unknown command: ", cmd)
