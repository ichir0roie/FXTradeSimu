def simple(values):
    vals = [(v[0] + v[1] + v[2] + v[3]) / 4 for v in values]
    ret = 0
    for i in vals:
        ret += i
    return ret / len(values)



