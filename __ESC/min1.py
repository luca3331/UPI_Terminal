import numpy as np
Tem_list = [[B, R, Y],
            [B, B, R],
            [R, R, Y]]
p1 = 'R'
p2 = 'B'
p3 = 'Y'

Puyo_Color = ["R", "B", "Y"]
search_list

for set_puyo_color in range(len(Puyo_Color)):
    for in_loop_y in range(3):
        for in_loop_x in range(3):
            if(Tem_list[y,x] == set_puyo_color):
                score = look_neighbor()



def check_of_bound(x,y):
    """参照するマスが盤面内かどうかの判定"""
    if(((x <= 0)or(x >= 7)) or ((y <= 0)or(y >= 14))):
        return False
    return True

def look_neighbor():
    coordinate_delta = [[-1,0],[0,+1],[+1,0],[0,-1]]
    for loop in range(len(coordinate_delta)):
        if ()
    look_coordinate
    pass