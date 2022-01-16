def head_tale_merge():
    head_arr = [0, 2, 2, 3, 3, 2, 0, 2, 0, 0, 0, 0]

    tale_arr = [0, 2, 2, 3, 3, 0, 2, 2, 0, 0, 0, 0]

    arr = []
    for num in range(2, len(head_arr), 3):
        arr.append(head_arr[num - 2])
        arr.append(head_arr[num - 1])
        arr.append(head_arr[num])
        arr.append(tale_arr[num - 2])
        arr.append(tale_arr[num - 1])
        arr.append(tale_arr[num])

    return arr

print(head_tale_merge())