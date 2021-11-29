def load_instance(file):
    with open(file) as f:
        n = int(f.readline())
        I = []
        for e in f.readline().split(" "):
            I.append(float(e))
        T = []
        for e in f.readline().split(" "):
            T.append(float(e))
        D = []
        for i in range(n):
            d = []
            for e in f.readline().split(" "):
                d.append(float(e))
            D.append(d)
    return n, I, T, D


class TapInstance:

    def __init__(self, file):
        n, I, T, D = load_instance(file)
        self.file = file
        self.size = n
        self.interest = I
        self.time = T
        self.dist = D

    def __repr__(self) -> str:
        return "TAP Instance of Size " + str(self.size) + " from '" + self.file + "'"

