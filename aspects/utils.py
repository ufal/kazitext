import numpy as np


def align(orig, cor):
    # Sentence lengths
    o_len = len(orig)
    c_len = len(cor)

    # Create the cost_matrix and the op_matrix
    cost_matrix = [[0.0 for j in range(c_len + 1)] for i in range(o_len + 1)]
    op_matrix = [["O" for j in range(c_len + 1)] for i in range(o_len + 1)]
    # Fill in the edges
    for i in range(1, o_len + 1):
        cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
        op_matrix[i][0] = "D"
    for j in range(1, c_len + 1):
        cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
        op_matrix[0][j] = "I"

    # Loop through the cost_matrix
    for i in range(o_len):
        for j in range(c_len):
            # Matches
            if orig[i] == cor[j]:
                cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                op_matrix[i + 1][j + 1] = "M"
            # Non-matches
            else:
                del_cost = cost_matrix[i][j + 1] + 1
                ins_cost = cost_matrix[i + 1][j] + 1
                # Standard Levenshtein (S = 1)
                sub_cost = cost_matrix[i][j] + 1

                # Transpositions require >=2 tokens
                # Traverse the diagonal while there is not a Match.
                if sorted(orig[i - 1:i + 1].lower()) == sorted(cor[j - 1:j + 1].lower()):
                    trans_cost = cost_matrix[i - 1][j - 1] + 1
                else:
                    trans_cost = float("inf")

                # Costs
                costs = [trans_cost, sub_cost, ins_cost, del_cost]
                # Get the index of the cheapest (first cheapest if tied)
                l = costs.index(min(costs))
                # Save the cost and the op in the matrices
                cost_matrix[i + 1][j + 1] = costs[l]
                if l == 0:
                    op_matrix[i + 1][j + 1] = "T"
                elif l == 1:
                    op_matrix[i + 1][j + 1] = "S"
                elif l == 2:
                    op_matrix[i + 1][j + 1] = "I"
                else:
                    op_matrix[i + 1][j + 1] = "D"
    # Return the matrices
    return cost_matrix, op_matrix


# Get the cheapest alignment sequence and indices from the op matrix
# align_seq = [(op, o_start, o_end, c_start, c_end), ...]
def get_cheapest_align_seq(orig, cor):
    _, op_matrix = align(orig, cor)
    i = len(op_matrix) - 1
    j = len(op_matrix[0]) - 1
    align_seq = []
    # Work backwards from bottom right until we hit top left
    while i + j != 0:
        # Get the edit operation in the current cell
        op = op_matrix[i][j]
        # Matches and substitutions
        if op in {"M", "S"}:
            align_seq.append((op, i - 1, i, j - 1, j))
            i -= 1
            j -= 1
        # Deletions
        elif op == "D":
            align_seq.append((op, i - 1, i, j, j))
            i -= 1
        # Insertions
        elif op == "I":
            align_seq.append((op, i, i, j - 1, j))
            j -= 1
        # Transpositions
        else:
            # Get the size of the transposition
            align_seq.append((op, i - 1, i, j - 1, j))
            i -= 1
            j -= 1
    # Reverse the list to go from left to right and return
    align_seq.reverse()
    return align_seq


def _apply_smoothing(unnormalized_probs, alpha, beta):
    '''
    :param unnormalized_probs: list with (un-normalized) "probability" values for each class. This list can also contain counts instead of
        probabilities, in this case, no alpha smoothing is done (as it would normalize the sum of these counts to 1).
    :param alpha: multiplication factor for all probabilities (how greater / smaller they will be). Note that if probabilities sum up to 1,
    :param beta: uniformity smoothing factor (if 1 new distribution is uniform, when 0 it is kept as is)
    :return:
    '''

    if len(unnormalized_probs) == 0:
        return []

    # first apply beta uniformity smoothing
    unnormalized_probs_sum = np.sum(unnormalized_probs)
    smoothed_unnormalized_probs = []
    for unnormalized_prob in unnormalized_probs:
        smoothed_value = unnormalized_prob * (1 - beta) + beta * (unnormalized_probs_sum / len(unnormalized_probs))
        smoothed_unnormalized_probs.append(smoothed_value)

    # then apply alpha multiplication (and potential clipping)
    # do not apply it when the sum is already 1 (full distribution)
    if not np.isclose(unnormalized_probs_sum, 1.):
        multiplication_factor = min(alpha, 1 / (unnormalized_probs_sum + 1e-6))
        for i in range(len(smoothed_unnormalized_probs)):
            smoothed_unnormalized_probs[i] *= multiplication_factor

    return smoothed_unnormalized_probs


def _apply_smoothing_on_simple_dict(simple_dict, alpha, beta):
    return {k: v for k, v in zip(simple_dict.keys(), _apply_smoothing(list(simple_dict.values()), alpha, beta))}
