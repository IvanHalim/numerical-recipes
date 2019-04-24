def readfile(file):
    with open(file) as f:
        array = [[float(x) for x in line.split(',')] for line in f]
    return array

