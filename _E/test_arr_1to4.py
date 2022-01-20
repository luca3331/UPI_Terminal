import numpy as np


def arr_1to4(arr):
    list_d1 = arr
    list_d4 = np.full((4, 6), -1)
    for lp in range(len(arr)):
        row = lp // 6
        col = lp % 6
        list_d4[row][col] = arr[lp]
    return list_d4

def arr_4to1(arr):
    list_d1 = [-1]*24
    list_d4 = arr
    for lp in range(24):
        list_d1[lp] = list_d4[lp // 6][lp % 6]
    return list_d1


list1 = [1, 2, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0]
print(arr_1to4(list1))
print(arr_4to1(arr_1to4(list1)))
