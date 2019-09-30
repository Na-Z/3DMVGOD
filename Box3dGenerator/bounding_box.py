import numpy as np


def create_AABB(ptcloud, label):
    '''
    create axis-aligned bounding box for the input point cloud
    :param ptcloud: np.ndarray with shape (n,3)
    :return: bbox3d, np.array with shape (8,), parameterized with [x, y, z, l, w, h, rotation_angle, label_id],
                                            where xyz represent center, l,w,h represent dimensions
    '''
    bbox = np.zeros((8))

    min_xyz = np.amin(ptcloud, axis=0)
    max_xyz = np.amax(ptcloud, axis=0)

    center = 0.5 * (min_xyz + max_xyz)
    dimensions = max_xyz - min_xyz

    bbox[0:3] = center
    bbox[3:6] = dimensions
    bbox[7] = label

    return bbox
