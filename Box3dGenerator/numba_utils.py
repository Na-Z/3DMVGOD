from numba import jit

@jit
def groupby_count(ptcloud, indices, out):
    for i in range(ptcloud.shape[0]):
        out[indices[i]] += 1
    return out


@jit
def groupby_sum(ptcloud, indices, out):
    for i in range(ptcloud.shape[0]):
        for j in range(4):
            out[indices[i], j] += ptcloud[i,j]
    return out
