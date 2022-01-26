"便宜上1次元リストをarr,2次元リストをlistと命名している"
import numpy as np


class Data:
    x_bound = 6
    y_bound = 4
    size_def = x_bound * y_bound
    list_temp = np.full((y_bound, x_bound), -1)  # 定形を4*6で表したリスト

    arr_temp = [[2, 2, 3, 4, 4, 7, 1, 1, 2, 3, 3, 4, 1, 2, 3, 8, 4, 9, 10, 11, 12, 0, 13, 0]]  # 定形を1*24で表したリスト
    arr_temp_bias = [[500, 500, 500, 500, 0, -999999, 100, 100, 100, 100, 100, 100, 100]]
    taboo_set = [[[1, 2], [1, 10], [2, 3], [2, 11], [3, 4], [3, 7], [3, 12], [4, 7], [4, 8], [4, 13]]]
    asc_mtx = np.full((10, 24, 24), -1)
    asc_mtx_score = np.full((10, 24, 24), -1)

    def __init__(self):
        pass

    def current(self, tier):
        """
        tierを受け取り，Data内の各種データを返却する
        :param tier:
        :return:
        """
        arr = self.arr_temp[tier]
        bias = self.arr_temp_bias[tier]
        taboo = self.taboo_set[tier]
        asc = self.asc_mtx[tier]
        asc_score = self.asc_mtx_score[tier]
        return arr, bias, taboo, asc, asc_score


class Asc:

    def __init__(self):
        self.list_temp = np.full((Data.y_bound, Data.x_bound), -1)  # 定形を4*6で表したリスト

    def asc_proc(self, tier):
        value = Data()
        value = Data.current(value, tier - 1)
        asc_ = self.make_asc_mtx(value[0], value[2], value[3])
        Data.asc_mtx[tier - 1] = asc_
        asc_score = self.asc_mtx_score(asc_, value[4], tier -1)
        Data.asc_mtx_score[tier - 1] = asc_score
        print(Data.asc_mtx[tier - 1], "\n")
        print(Data.asc_mtx_score[tier - 1])

    def make_asc_mtx(self, arr, taboo, asc):
        """
        定形の状態行列を返却する
        :param arr:
        :param taboo:
        :param asc:
        :return:
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
                if asc[row][col] == 0:
                    asc_score[row][col] = 0
                elif asc[row][col] == 1:
                    asc_score[row][col] = (arr_bias[arr_temp[row - 1]] + arr_bias[arr_temp[col - 1]]) / 2
                elif asc[row][col] == -1:
                    asc_score[row][col] = (arr_bias[arr_temp[row - 1]] + arr_bias[arr_temp[col - 1]]) / 2 * -1
                else:
                    print("ex I am", row, col, asc[row][col])
        return asc_score


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


if __name__ == "__main__":
    ob = Asc()
    ob.asc_proc(1)

arr_temp = [[2, 2, 3, 4, 4, 7, 1, 1, 2, 3, 3, 4, 1, 2, 3, 8, 4, 9, 10, 11, 12, 0, 13, 0],
                [2, 2, 3, 4, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 7, 4, 8, 9, 10, 11, 0, 12, 0],
                [2, 2, 3, 7, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 8, 4, 4, 9, 10, 11, 0, 12, 13],
                [2, 2, 3, 4, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 7, 4, 4, 8, 9, 10, 0, 11, 12],
                [2, 2, 3, 7, 4, 4, 1, 1, 2, 3, 3, 4, 1, 2, 3, 8, 4, 9, 10, 11, 12, 0, 13, 0],
                [2, 2, 3, 7, 8, 9, 1, 1, 2, 3, 3, 4, 1, 2, 3, 10, 11, 4, 12, 13, 14, 4, 4, 15],
                [2, 2, 3, 7, 8, 4, 1, 1, 2, 3, 3, 4, 1, 2, 3, 4, 4, 9, 10, 11, 12, 13, 14, 15, 0],
                [2, 2, 7, 4, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 8, 4, 9, 10, 11, 12, 13, 0, 14, 0]]  # 定形を1*24で表したリスト
    arr_temp_bias = [[500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                     [500, 500, 500, 500, 1000, -999999, 100, 100, 100, 100, 100, 100, 100, 100]]
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
