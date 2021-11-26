def heuristic(weights, val, capa):
    heuris = [0] * len(weights)

    elems = [(i, val[i]/weights[i]) for i in range(len(weights))]
    elems = sorted(elems, key=lambda x: x[1])
    elems.reverse()

    current = 0
    for e in elems:
        if current + weights[e[0]] <= capa:
            heuris[e[0]] = 1
            current += weights[e[0]]
    return heuris