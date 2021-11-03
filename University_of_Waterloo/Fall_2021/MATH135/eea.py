import numpy as np

def EEA(x: int, y: int) -> np.ndarray:
    """Generates a numpy array of an Extended Euclidean Algorithm table.

    Parameters
    ----------
    x
        Must be greater than 0. Order doesn't matter
    y
        Must be greater than 0. Order doesn't matter
    """
    a = max(x, y)
    b = min(x, y)

    # initial rows for the EEA table.
    r1 = np.array([1, 0, a, 0], dtype=int)
    r2 = np.array([0, 1, b, 0], dtype=int)
    big_arr = np.vstack((r1, r2))
    while r2[2] > 0:
        # r2 will be changing over time.

        q = r1[2] // r2[2]

        # subtract the first 3 elements of r1  from the first three elements
        # of r2 * q

        r3 = r1[:3] - q * r2[:3]

        # update the rows for the next iteration
        r1 = r2
        r2 = r3

        # must append because q r3 only contains 3 values.
        r2 = np.append(r2, q)

        # stack r2 on top of the big array we're building.
        big_arr = np.vstack((big_arr, r2))

    return big_arr


def latex_EEA(x: int, y: int) -> str:
    """Generates a string for a latex table based on the Extended Euclidean Algorithm."""
    EEA_table = EEA(x, y)
    building_string = r"""
\begin{tabular}{|c|c|c|c|}
    \hline
    $x$&$y$&$r$&$q$\\ \hline
"""
    for row in EEA_table:
        # mimicking tab spacing
        building_string += f"""    {row[0]}&{row[1]}&{row[2]}&{row[3]}"""
        building_string += r"""\\ \hline
"""
    building_string += r"""\end{tabular}"""
    return building_string




