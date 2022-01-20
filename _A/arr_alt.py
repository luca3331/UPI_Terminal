import numpy as np

def_arr = [[1, 2, 3, 0, 4, 0],
           [1, 1, 2, 3, 3, 4],
           [2, 2, 0, 4, 3, 4]]

tri_list = np.full((18, 18), -1)


def is_next_to(row, col, lp1, lp2):
    if row == lp1 - 1 and col == lp2:
        return True
    elif row == lp1 + 1 and col == lp2:
        return True
    elif row == lp1 and col == lp2 - 1:
        return True
    elif row == lp1 and col == lp2 + 1:
        return True
    else:
        return False


def make_tri_list():
    for row in range(3):
        for col in range(6):
            color = def_arr[row][col]

            for lp1 in range(row, 3):
                for lp2 in range(col, 6):
                    if color == 0 or def_arr[lp1][lp2] == 0:
                        tri_list[row * 6 + col][lp1 * 6 + lp2] = 0
                        continue
                    elif color == def_arr[lp1][lp2]:
                        tri_list[row * 6 + col][lp1 * 6 + lp2] = 1
                        continue
                    elif color != def_arr[lp1][lp2]:
                        # if is_next_to(row, col, lp1, lp2):
                        tri_list[row * 6 + col][lp1 * 6 + lp2] = -1
                        # else:
                        #     tri_list[row * 6 + col][lp1 * 6 + lp2] = 0
                    else:
                        pass
    return tri_list


print(make_tri_list())
