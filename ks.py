def heuristic(weights, val, distances, et, ed):
    sol = []

    elems = [(i, val[i]/weights[i]) for i in range(len(weights))]
    elems = sorted(elems, key=lambda x: x[1])
    elems.reverse()
    elems = [e[0] for e in elems]

    current_w = 0
    current_d = 0

    for e in elems:
        if current_w + weights[e] <= et:
            if len(sol) == 0:
                sol.append(e)
                current_w += weights[e]
            else:
                bi, bi_cost = __best_insert(sol, distances, e)
                if current_d + bi_cost < ed:
                    current_d += bi_cost
                    current_w += weights[e]
                    # cool now do the insert
                    if bi == 0:
                        sol = [e] + sol
                    elif bi == len(sol):
                        sol.append(e)
                    else:
                        sol = sol[:bi] + [e] + sol[bi:]

    return sol


def __best_insert(sol, distances, qid):
    # check fisrt
    best = (0, distances[qid][sol[0]])
    # check last
    if distances[sol[len(sol)-1]][qid] < best[1]:
        best = (len(sol), distances[sol[len(sol)-1]][qid])
    # check every other place
    for i in range(1, len(sol)):
        # insert before query at pos i
        d = distances[sol[i-1]][qid] + distances[qid][sol[i]] - distances[sol[i-1]][sol[i]]
        if d < best[1]:
            best = (i, d)

    return best