import numpy as np

def _class_to_index(mask):
    key = np.array([-1, -1, -1, -1, -1, -1,
                          -1, -1, 0, 1, -1, -1,
                          2, 3, 4, -1, -1, -1,
                          5, -1, 6, 7, 8, 9,
                          10, 11, 12, 13, 14, 15,
                          -1, -1, 16, 17, 18])
    mapping = np.array(range(-1, len(key) - 1)).astype('int32')

    values = np.unique(mask)
    for value in values:
        assert (value in mapping)
    index = np.digitize(mask.ravel(), mapping, right=True)
    return key[index].reshape(mask.shape)


if __name__ == '__main__':
    # mask = np.random.randint(7,22,(3,3))
    mask = np.full(3,0)
    print(mask)
    print(_class_to_index(mask))
    nu = np.array([1,2,3,45,5])
