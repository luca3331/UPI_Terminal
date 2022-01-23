SIZE = 128
tumo_tmp = [''] * 65536
tumo = ['tumo'] * 65536
flread = open('haipuyo.txt', 'r')
count = 0

def tumo_edit(text):
    out_list = [' '] * 64
    for lp in range(0, 128, 2):
        out_list.insert(lp + 1, text[lp:lp+2])
    out_txt = ''.join(out_list)
    return out_txt

for line in flread:
    tumo_tmp[count] = line + "\n"
    tumo[count] = 'tumo' + tumo_edit(tumo_tmp[count])
    # filename = 'tumo' + str(count) + '.txt'
    # flwrite = open(filename, 'w')
    # flwrite.write(tumo_edit(tumo[count]))
    count += 1
    # if count > 100:
    #     print(tumo)
    #     exit()

print(tumo)