import numpy as np
from numpy import dot, inner, array, linalg

def quat2rot(quats):
    quats = array(quats)
    input_shape = quats.shape
    quats = np.atleast_2d(quats)
    Rs = np.zeros((quats.shape[0], 3, 3))
    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]
    Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
    Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
    Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    if len(input_shape) < 2:
        return Rs[0]
    else:
        return Rs

def euler2quat(eulers):
    eulers = array(eulers)
    if len(eulers.shape) > 1:
        output_shape = (-1,4)
    else:
        output_shape = (4,)
    eulers = np.atleast_2d(eulers)
    gamma, theta, psi = eulers[:,0],  eulers[:,1],  eulers[:,2]

    q0 = np.cos(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) + \
       np.sin(gamma / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    q1 = np.sin(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) - \
       np.cos(gamma / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    q2 = np.cos(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2) + \
       np.sin(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    q3 = np.cos(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2) - \
       np.sin(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2)

    quats = array([q0, q1, q2, q3]).T
    for i in range(len(quats)):
        if quats[i,0] < 0:
            quats[i] = -quats[i]
    return quats.reshape(output_shape)

def euler2rot(eulers):
    return quat2rot(euler2quat(eulers))

device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T

MODEL_INPUT_SIZE = (256, 128)
MODEL_YUV_SIZE = (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1] * 3 // 2)
MODEL_CX = MODEL_INPUT_SIZE[0]/2.
MODEL_CY = 21.

model_zoom = 1.25
model_height = 1.22
eon_focal_length = FOCAL = 910.0

# canonical model transform
model_intrinsics = np.array(
  [[ eon_focal_length / model_zoom,    0. ,  MODEL_CX],
   [   0. ,  eon_focal_length / model_zoom,  MODEL_CY],
   [   0. ,                            0. ,   1.]])

def get_view_frame_from_road_frame(roll, pitch, yaw, height):
    device_from_road = euler2rot([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
    view_from_road = view_frame_from_device_frame.dot(device_from_road)
    return np.hstack((view_from_road, [[0], [height], [0]]))