"""
input;n*mサイズの１次元リスト
output:(n*m)*(n*m)の行列

行と列を入力
row_def = input()
col_def = input()
どうせ列は6なので，input数から6を割ったら行もわかる
"""
import numpy as np

row_def = 4
col_def = 6
size_def = row_def * col_def
arr = np.full((1, size_def), -1)
Temp_arr = np.full((size_def, size_def), -1)
Temp_score = np.full((size_def, size_def), -1)

arr = [2, 2, 0, 0, 4, 4, 1, 1, 2, 3, 3, 3, 1, 2, 3, 0, 0, 4, 4, 4, 0, 0, 0, 0]  # とりあえず手入力
Temp_score = [0, 320, 200, 100, 50]  # 1,2,3,4のそれぞれの重み評価値
arr_score_added = [-1]*24

for lp1 in range(size_def):
    arr_score_added[lp1] = Temp_score[arr[lp1]]

for lp1 in range(size_def):
    for lp2 in range(size_def):
        if (arr[lp1] == 0) or arr[lp2] == 0:
            Temp_arr[lp1][lp2] = 0
            continue
        if arr[lp1] == arr[lp2]:
            Temp_arr[lp1][lp2] = 1
            continue
        if arr[lp1] != arr[lp2]:
            Temp_arr[lp1][lp2] = -1
            continue


for lp1 in range(size_def):
    for lp2 in range(size_def):
        if arr[lp1] == arr[lp2]:
            Temp_arr[lp1][lp2] = (arr_score_added[lp1] + arr_score_added[lp2]) / 2
            continue
        if arr[lp1] != arr[lp2]:
            Temp_arr[lp1][lp2] = (arr_score_added[lp1] + arr_score_added[lp2]) / 2 * -1
            continue

print(Temp_arr)

