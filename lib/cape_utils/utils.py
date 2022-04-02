import numpy as np
import os
import glob
from plyfile import PlyData
import time, re

# joint names of SMPL
J_names = { 0: 'Pelvis',
    1: 'L_Hip',
    4: 'L_Knee',
    7: 'L_Ankle',
    10: 'L_Foot',

    2: 'R_Hip',
    5: 'R_Knee',
    8: 'R_Ankle',
    11: 'R_Foot',

    3: 'Spine1',
    6: 'Spine2',
    9: 'Spine3',
    12: 'Neck',
    15: 'Head',

    13: 'L_Collar',
    16: 'L_Shoulder',
    18: 'L_Elbow',
    20: 'L_Wrist',
    22: 'L_Hand',
    14: 'R_Collar',
    17: 'R_Shoulder',
    19: 'R_Elbow',
    21: 'R_Wrist',
    23: 'R_Hand',
}

# indices of SMPL joints related to clothing
useful_joints_idx = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19]

def filter_cloth_pose(pose_vec):
    '''
    Remove SMPL pose params from clothing-unrelated joints
    args:
        pose_vec: flattened 72-dim pose vector or 216-dim rotational matrix
    returns:
        42(=14*3)-dim pose vector or 126(=14*9) dim rot matrix that is relevant to clothing
    '''
    num_examples = pose_vec.shape[0]
    dim = pose_vec.shape[-1]

    # 24x3, SMPL pose parameters
    if dim == 72:
        pose_array = pose_vec.reshape(num_examples, -1, 3)

    # 24x9, rotational matrices of the pose parameters
    elif dim == 216:
        pose_array = pose_vec.reshape(num_examples, -1, 9)
    else:
        print('please provide either 72-dim pose vector or 216-dim rot matrix')
        return
    useful_pose_array = pose_array[:, useful_joints_idx, :]
    return useful_pose_array.reshape(num_examples, -1)

def row(A):
    return tf.reshape(A, (1, -1))

def col(A):
    return tf.reshape(A, (-1, 1))

def sparse2tfsparse(sparse_matrix):
    '''
    turn a scipy sparse csr_matrix into a tensorflow sparse matrix
    '''
    sparse_matrix = sparse_matrix.tocoo()
    indices = np.column_stack((sparse_matrix.row, sparse_matrix.col))
    sparse_matrix = tf.SparseTensor(indices, sparse_matrix.data, sparse_matrix.shape)
    sparse_matrix = tf.sparse_reorder(sparse_matrix)
    return sparse_matrix

def pose2rot(pose):
    '''
    use Rodrigues transformation to turn pose vector to rotation matrix
    args:
        pose: [num_examples, 72], unraveled version of the pose vector in axis angle representation (24*3)
    returns:
        rot_all: rot matrix, [num_examples, 216], (216=24*3*3)
    '''
    from cv2 import Rodrigues
    num_examples = pose.shape[0]
    pose = pose.reshape(num_examples, -1, 3)

    rot_all = [np.array([Rodrigues(pp[i, :])[0] for i in range(pp.shape[0])]).ravel() for pp in pose]
    rot_all = np.array(rot_all)
    return rot_all

def rot2pose(rot):
    '''
    use Rodrigues transformation to turn rotation matrices into pose vector
    args:
        rot: [num_examples, 216], unraveled version of the 3x3 rot matrix (216=24 joints * 3*3)
    returns:
        pose_vec: pose vector [num_examples, 72], pose vector in axis angle representation (72=24*3)
    '''
    from cv2 import Rodrigues
    num_examples = rot.shape[0]
    rot = rot.reshape(num_examples, -1, 9)
    pose_vec = [np.array([Rodrigues(rr[i, :].reshape(3,3))[0] for i in range(rr.shape[0])]).ravel() for rr in rot]
    pose_vec = np.array(pose_vec)

    return pose_vec


'''

Following: TensorFlow implementation of psbody.mesh.geometry.tri_normals

'''

def TriNormals(v, f):
    return NormalizedNx3(TriNormalsScaled(v, f))

def TriNormalsScaled(v, f):
    edge_vec1 = tf.reshape(TriEdges(v, f, 1, 0), (-1, 3))
    edge_vec2 = tf.reshape(TriEdges(v, f, 2, 0), (-1, 3))
    return tf.cross(edge_vec1, edge_vec2)

def NormalizedNx3(v):
    v = tf.reshape(v, (-1, 3))
    ss = tf.reduce_sum(tf.square(v), axis=1)
    # prevent zero division
    indices = tf.equal(ss, 0.)
    mask = tf.cast(indices, ss.dtype)  # a mask, 1 where norms==0, 0 otherwise
    norms = tf.add(ss, mask)
    s = tf.sqrt(norms)
    return tf.reshape(tf.divide(v, col(s)), [-1])

def TriEdges(v, f, cplus, cminus):
    assert(cplus >= 0 and cplus <= 2 and cminus >= 0 and cminus <= 2)
    return _edges_for(v, f, cplus, cminus)

def _edges_for(v, f, cplus, cminus):
    # return (
    #     v.reshape(-1, 3)[f[:, cplus], :] -
    #     v.reshape(-1, 3)[f[:, cminus], :]).ravel()

    ind_plus, ind_minus = f[:, cplus], f[:, cminus]
    ind_plus = tf.expand_dims(ind_plus, 1)
    ind_minus = tf.expand_dims(ind_minus, 1)
    v = tf.reshape(v, (-1, 3))

    t = tf.gather_nd(v, ind_plus) - tf.gather_nd(v, ind_minus)
    return tf.reshape(t, [-1])


def read_point_ply(filename):
    """Load point cloud from ply file.

    Args:
      filename: str, filename for ply file to load.
    Returns:
      v: np.array of shape [#v, 3], vertex coordinates
      n: np.array of shape [#v, 3], vertex normals
    """
    pd = PlyData.read(filename)['vertex']
    try:
        v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
        n = np.array(np.stack([pd[i] for i in ['nx', 'ny', 'nz']], axis=-1))
    except:
        v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
        n = np.zeros_like(v).astype(np.float32)
    return v, n

def load_input_ply(ply_folder):
    files = sorted(glob.glob(os.path.join(ply_folder, '*.ply')))
    vs = []
    vns = []
    n_max = 0
    for f in files:
        v, n = read_point_ply(f)
        v = v.astype(np.float32)
        n = n.astype(np.float32)
        vs.append(v)
        vns.append(n)
        n_max = max(v.shape[0], n_max)
    vs = np.array(vs)
    vns = np.array(vns)
    return vs, vns, n_max

def sample_points_from_ray(points, normals, sample_factor=10, std=0.01):
    """Get sample points from points from ray.

    Args:
      points (numpy array): [npts, 3], xyz coordinate of points on the mesh surface.
      normals (numpy array): [npts, 3], normals of points on the mesh surface.
      sample_factor (int): number of samples to pick per surface point.
      std (float): std of samples to generate.
    Returns:
      points (numpy array): [npts*sample_factor, 3], where last dimension is
      distance to surface point.
      sdf_values (numpy array): [npts*sample_factor, 1], sdf values of the sampled points
      near the mesh surface.
    """
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    npoints = points.shape[0]
    offsets = np.random.randn(npoints, sample_factor, 1) * std
    point_samples = points[:, np.newaxis, :] + offsets * normals[:, np.newaxis, :]
    point_samples = point_samples.reshape(-1, points.shape[1])
    sdf_values = offsets.reshape(-1, 1)
    point_samples = point_samples.astype(np.float32)
    sdf_values = sdf_values.astype(np.float32)
    return point_samples, sdf_values


def np_pad_points_occ(points, occ, ntarget):
    """Pad point cloud to required size.

    If number of points is larger than ntarget, take ntarget random samples.
    If number of points is smaller than ntarget, pad by repeating last point.
    Args:
      points: `[npoints, nchannel]` np array, where first 3 channels are xyz.
      ntarget: int, number of target channels.
    Returns:
      result: `[ntarget, nchannel]` np array, padded points to ntarget numbers.
    """
    if points.shape[0] < ntarget:
        mult = np.ceil(float(ntarget) / float(points.shape[0])) - 1
        rand_pool = np.tile(points, [int(mult), 1])
        rand_pool_occ = np.tile(occ, [int(mult), 1])
        nextra = ntarget - points.shape[0]
        extra_idx = np.random.choice(rand_pool.shape[0], nextra, replace=False)
        extra_pts = rand_pool[extra_idx]
        extra_occ = rand_pool_occ[extra_idx]
        points_out = np.concatenate([points, extra_pts], axis=0)
        occ_out = np.concatenate([occ, extra_occ], axis=0)
    else:
        idx_choice = np.random.choice(points.shape[0], size=ntarget, replace=False)
        points_out = points[idx_choice]
        occ_out = occ[idx_choice]

    return points_out, occ_out.squeeze()