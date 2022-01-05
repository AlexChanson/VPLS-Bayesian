def get_solution_as_sequence(cplex_sol, ist_size):
    """
    Converts cplex solution to a usable version
    (Dev Warning: this method uses offset id to account for fictitious start and end queries)
    :param cplex_sol: the cples solver solution object
    :param ist_size: size of the problem instance
    :return: the solution as an ordered list of query indexes matching the instance
    """
    start = 0
    for i in range(0, ist_size+2):
        if cplex_sol.get_value('x_0_' + str(i)):
            start = i

    s = [start]
    prev = start
    while prev != ist_size + 1:
        for i in range(1, ist_size + 2):
            if cplex_sol.get_value('x_' + str(prev) + '_' + str(i)):
                prev = i
                s.append(i)
                break
    # Convert back to user ids before returning
    return list(map(lambda x: x - 1, s[:-1]))