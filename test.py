import os
import pickle

import cv2

import ai2thor.controller

import numpy as np

from enum import Enum
from scipy.spatial.transform import Rotation as R
from math import cos, sin, radians

from sklearn.externals import joblib


class CameraArrayAlongAxis(Enum):
    X = 0
    Z = 1


class Camera:
    def __init__(self, location, rotation):
        """ Init camera params

        :location : (x, y, z) location of the camera
        :rotation : (rx, ry, rz) angles of the camera
        """
        self.location = location
        self.rotation = rotation


def compute_rotation_matrix(rx, ry, rz):
    rrx = radians(rx)
    rry = radians(ry)
    rrz = radians(rz)

    Rx = np.array([[1, 0, 0], [0, cos(rrx), -sin(rrx)], [0, sin(rrx), cos(rrx)]])
    Ry = np.array([[cos(rry), 0, sin(rry)], [0, 1, 0], [-sin(rry), 0, cos(rry)]])
    Rz = np.array([[cos(rrz), -sin(rrz), 0], [sin(rrz), cos(rrz), 0], [0, 0, 1]])

    return Rx @ Ry @ Rz

def write_frame(out_path, scene, controller, idx):
    scene_dict = {'name': scene, 0: {}, 90: {}, 180: {}, 270: {}}

    print('Name: ', scene)
    controller.reset(scene=scene)
    
    event = controller.step(dict(action='Initialize'), renderDepthImage=True, gridSize=0.25)
    
    corners = np.array(event.metadata['sceneBounds']['cornerPoints'])
    min_x, max_x = round(np.min(corners[:, 0])), round(np.max(corners[:, 0]))
    min_z, max_z = round(np.min(corners[:, 2])), round(np.max(corners[:, 2]))

    for ry in (0, 90, 180, 270):
        for j in np.arange(0.75, 1.21, 0.25):
            for k in np.arange(min_z, max_z + 0.1, 0.25):
                for i in np.arange(min_x, max_x + 0.1, 0.25):
                    if ry == 0 and k + 1.5 > max_z:
                        continue

                    if ry == 90 and i + 1.5 > max_x:
                        continue

                    if ry == 180 and k - 1.5 < min_z:
                        continue

                    if ry == 270 and i - 1.5 < min_x:
                        continue

                    if (ry == 0 or ry == 180) and (i > max_x - 1.5 or i < min_x + 1.5):
                        continue

                    if (ry == 90 or ry == 270) and (k > max_z - 1.5 or k < min_z + 1.5):
                        continue

                    event = controller.step(action='TeleportFull', x=i, y=j, z=k, rotation=dict(x=0, y=ry, z=0), horizon=0.0)

                    position = event.metadata['agent']['position']
                    x, y, z = position['x'], position['y'], position['z']

                    if abs(i-x) + abs(j - y) + abs(k - k) > 0.05:
                        continue

                    print(i, j, k, x, y, z, ' --- ', ry, scene)

                    r1 = event.metadata['agent']['r1']
                    r2 = event.metadata['agent']['r2']
                    r3 = event.metadata['agent']['r3']
                    t = event.metadata['agent']['t']

                    p1 = event.metadata['agent']['p1']
                    p2 = event.metadata['agent']['p2']
                    p3 = event.metadata['agent']['p3']
                    p4 = event.metadata['agent']['p4']

                    rp1 = event.metadata['agent']['rp1']
                    rp2 = event.metadata['agent']['rp2']
                    rp3 = event.metadata['agent']['rp3']
                    rp4 = event.metadata['agent']['rp4']

                    pj1 = event.metadata['agent']['pj1']
                    pj2 = event.metadata['agent']['pj2']
                    pj3 = event.metadata['agent']['pj3']
                    pj4 = event.metadata['agent']['pj4']
                    
                    pm = np.zeros((4, 4))

                    vp = event.metadata['agent']['vp']

                    image = event.frame
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                    depth = event.depth_frame
                    print(depth.shape)

                    norm_depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())
                    norm_depth = norm_depth.astype(np.uint8)

                    extrinsic = np.zeros((4, 4))

                    extrinsic[0, 0] = r1['x']
                    extrinsic[0, 1] = r1['y']
                    extrinsic[0, 2] = r1['z']
                    extrinsic[1, 0] = r2['x']
                    extrinsic[1, 1] = r2['y']
                    extrinsic[1, 2] = r2['z']
                    extrinsic[2, 0] = r3['x']
                    extrinsic[2, 1] = r3['y']
                    extrinsic[2, 2] = r3['z']
                    extrinsic[0, 3] = t['x'] * 1000
                    extrinsic[1, 3] = y * 1000
                    extrinsic[2, 3] = t['z'] * 1000
                    extrinsic[3, 3] = 1

                    pm[0, 0] = pj1['x']
                    pm[0, 1] = pj1['y']
                    pm[0, 2] = pj1['z']
                    pm[0, 3] = pj1['w']
                    pm[1, 0] = pj2['x']
                    pm[1, 1] = pj2['y']
                    pm[1, 2] = pj2['z']
                    pm[1, 3] = pj2['w']
                    pm[2, 0] = pj3['x']
                    pm[2, 1] = pj3['y']
                    pm[2, 2] = pj3['z']
                    pm[2, 3] = pj3['w']
                    pm[3, 0] = pj4['x']
                    pm[3, 1] = pj4['y']
                    pm[3, 2] = pj4['z']
                    pm[3, 3] = pj4['w']

                    p = pm @ extrinsic

                    extrinsic[2, 0] = p[3, 0]
                    extrinsic[2, 1] = p[3, 1]
                    extrinsic[2, 2] = p[3, 2]
                    extrinsic[2, 3] = p[3, 3]

                    # print(depth[70, 70], depth[150, 150], depth[70, 150], depth[150, 70], depth[100, 100])
                    # print(extrinsic)
                    # exit()
                    # print(pm)
                    # print(vp)
                    # print(p1, p2, p3, p4)
                    # print(rp1, rp2, rp3, rp4)
                    
                    # cv2.imshow('depth', norm_depth)
                    # cv2.imshow('image', image)
                    # cv2.waitKey()

                    if ry == 0 or ry == 180:
                        if z not in scene_dict[ry].keys():
                            scene_dict[ry][z] = dict()
                            scene_dict[ry][z]['count'] = 0

                        if y not in scene_dict[ry][z].keys():
                            scene_dict[ry][z][y] = dict()

                        scene_dict[ry][z]['count'] += 1
                        scene_dict[ry][z][y][x] = {'x': x, 'y': y, 'z': z, 'depth': depth, 'image': image, 'extrinsic': extrinsic}
                    else:
                        if x not in scene_dict[ry].keys():
                            scene_dict[ry][x] = dict()
                            scene_dict[ry][x]['count'] = 0

                        if y not in scene_dict[ry][x].keys():
                            scene_dict[ry][x][y] = dict()

                        scene_dict[ry][x]['count'] += 1
                        scene_dict[ry][x][y][z] = {'x': x, 'y': y, 'z': z, 'depth': depth, 'image': image, 'extrinsic': extrinsic}

    # TO REMOVE REDUNDANT CAMERA ALONG AXIS X
    for ry in (0, 90, 180, 270):
        to_remove = []
        for i in scene_dict[ry].keys():
            if scene_dict[ry][i]['count'] <= 1:
                to_remove.append(i)
                
        for k in to_remove:
            scene_dict[ry].pop(i, None)

    for ry in (0, 90, 180, 270):
        for x in scene_dict[ry].keys():
            pickle.dump({'rotation': ry, 'data': scene_dict[ry][x], 'name': scene}, open(os.path.join(out_path, scene + '_' + str(ry) + '_' + str(x)), 'wb'))


# controller = ai2thor.controller.Controller(local_executable_path='/scratch/antruong/workspace/ai2thor/unity/builds/thor-local-Linux64')
def aithor_handling(out_path,
                    local_executable_path='/scratch/antruong/workspace/ai2thor/unity/builds/thor-local-Linux64'):
    controller = ai2thor.controller.Controller(local_executable_path=local_executable_path)
    controller.step(dict(action='ChangeResolution', x=300, y=300), raise_for_failure=True)

    # scenes = define_physic_scene_dict()
    # scenes.extend(define_train_scene_dict())

    # scenes = define_to_check_dict()

    scenes = get_list_scenes()
    for i, s in enumerate(scenes):
        write_frame(out_path, s, controller, i)


def get_list_scenes():
    scenes = ['FloorPlan1_physics', 'FloorPlan2_physics', 'FloorPlan3_physics', 'FloorPlan4_physics', 'FloorPlan5_physics',
              'FloorPlan6_physics', 'FloorPlan7_physics', 'FloorPlan8_physics', 'FloorPlan9_physics', 'FloorPlan10_physics',
              'FloorPlan11_physics', 'FloorPlan12_physics', 'FloorPlan13_physics', 'FloorPlan14_physics',
              'FloorPlan15_physics', 'FloorPlan16_physics', 'FloorPlan17_physics', 'FloorPlan18_physics',
              'FloorPlan19_physics', 'FloorPlan20_physics', 'FloorPlan21_physics', 'FloorPlan22_physics',
              'FloorPlan23_physics', 'FloorPlan24_physics', 'FloorPlan25_physics', 'FloorPlan26_physics',
              'FloorPlan27_physics', 'FloorPlan28_physics', 'FloorPlan29_physics', 'FloorPlan30_physics',
              'FloorPlan201_physics', 'FloorPlan202_physics', 'FloorPlan203_physics', 'FloorPlan204_physics',
              'FloorPlan205_physics', 'FloorPlan206_physics', 'FloorPlan207_physics', 'FloorPlan208_physics',
              'FloorPlan209_physics', 'FloorPlan210_physics', 'FloorPlan211_physics', 'FloorPlan212_physics',
              'FloorPlan213_physics', 'FloorPlan214_physics', 'FloorPlan215_physics', 'FloorPlan216_physics',
              'FloorPlan217_physics', 'FloorPlan219_physics', 'FloorPlan220_physics',
              'FloorPlan221_physics', 'FloorPlan222_physics', 'FloorPlan223_physics', 'FloorPlan224_physics',
              'FloorPlan225_physics', 'FloorPlan226_physics', 'FloorPlan227_physics', 'FloorPlan228_physics',
              'FloorPlan229_physics', 'FloorPlan230_physics']
    return scenes


def define_train_scene_dict():
    # 'FloorPlan_Train10_1
    scene_train_10_1_1 = {'name': 'FloorPlan_Train10_1', 'range_x': (3.0, 3.5), 'range_y': (0.5, 1.0),
                          'range_z': (-4.5, -1.0), 'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_train_10_1_2 = {'name': 'FloorPlan_Train10_1', 'range_x': (5.5, 6.0), 'range_y': (0.5, 1.0),
                          'range_z': (-3.5, -2.5), 'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_train_10_1_3 = {'name': 'FloorPlan_Train10_1', 'range_x': (2.0, 3.0), 'range_y': (0.5, 1.0),
                          'range_z': (-1.5, -1.0), 'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_train_10_1_4 = {'name': 'FloorPlan_Train10_1', 'range_x': (7.75, 8.25), 'range_y': (0.5, 1.0),
                          'range_z': (-1.5, -1.0), 'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scenes = [scene_train_10_1_1, scene_train_10_1_2, scene_train_10_1_3, scene_train_10_1_4]
    return scenes


def define_to_check_dict():
    # controller.reset(scene='FloorPlan8_physics')
    # event = controller.step(action='TeleportFull', x=-2.5, y=0.9, z=-1.0, rotation=dict(x=0, y=90, z=0), horizon=0.0)
    scenes = []
    return scenes

def define_physic_scene_dict():
    # 'FloorPlan1_physics
    scene_physics_11 = {'name': 'FloorPlan1_physics', 'range_x': (1.5, 2.0),  # close - far
                        'range_y': (0.5, 1.0), 'range_z': (-1.5, 1.5),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_12 = {'name': 'FloorPlan1_physics',
                        'range_x': (-1.5, 1.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.5, 2.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_13 = {'name': 'FloorPlan1_physics',
                        'range_x': (-1.5, -1.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.5, 1.5),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_14 = {'name': 'FloorPlan1_physics',
                        'range_x': (-1.5, 1.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-2.0, -1.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan2_physics
    scene_physics_21 = {'name': 'FloorPlan2_physics',
                        'range_x': (1.0, 1.25),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-0.5, 1.5),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_22 = {'name': 'FloorPlan2_physics',
                        'range_x': (-1.0, 1.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.5, 2.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_23 = {'name': 'FloorPlan2_physics',
                        'range_x': (-1.25, -1.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-0.5, 1.5),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_24 = {'name': 'FloorPlan2_physics',
                        'range_x': (-1.0, 1.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.0, -0.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan3_physics
    scene_physics_31 = {'name': 'FloorPlan3_physics',
                        'range_x': (-0.5, 0.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-2.0, 2.0),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_32 = {'name': 'FloorPlan3_physics',
                        'range_x': (-1.0, 0.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.5, 2.5),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_33 = {'name': 'FloorPlan3_physics',
                        'range_x': (-1.0, -0.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-2.0, 2.0),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_34 = {'name': 'FloorPlan3_physics',
                        'range_x': (-1.0, 0.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-2.5, -1.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan4_physics
    scene_physics_40 = {'name': 'FloorPlan4_physics',
                        'range_x': (-2.0, -1.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (1.0, 2.5),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_41 = {'name': 'FloorPlan4_physics',
                        'range_x': (-1.5, -0.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (1.0, 1.5),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_42 = {'name': 'FloorPlan4_physics',
                        'range_x': (-4.0, 0.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.5, 2.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_43 = {'name': 'FloorPlan4_physics',
                        'range_x': (-4.0, -1.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (1.0, 1.5),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_44 = {'name': 'FloorPlan4_physics',
                        'range_x': (-4.0, 0.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.0, 1.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan5_physics
    scene_physics_51 = {'name': 'FloorPlan5_physics',
                        'range_x': (0.75, 1.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.5, 0.5),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_52 = {'name': 'FloorPlan5_physics',
                        'range_x': (-1.0, 2.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.5, 2.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_53 = {'name': 'FloorPlan5_physics',
                        'range_x': (-0.5, -0.25),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.5, -0.5),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_54 = {'name': 'FloorPlan5_physics',
                        'range_x': (0.25, 1.25),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 0.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan6_physics
    scene_physics_60 = {'name': 'FloorPlan6_physics',
                        'range_x': (-0.25, 0.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.75, 0.0),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_61 = {'name': 'FloorPlan6_physics',
                        'range_x': (-0.75, -0.25),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.75, 2.0),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_62 = {'name': 'FloorPlan6_physics',
                        'range_x': (-1.25, 0.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (1.75, 2.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_63 = {'name': 'FloorPlan6_physics',
                        'range_x': (-1.25, -1.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.75, 2.0),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_64 = {'name': 'FloorPlan6_physics',
                        'range_x': (-1.25, 0.0),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.75, -1.25),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan7_physics
    scene_physics_70 = {'name': 'FloorPlan7_physics',
                        'range_x': (0.0, 1.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 1.5),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_71 = {'name': 'FloorPlan7_physics',
                        'range_x': (1.0, 2.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 2.0),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_72 = {'name': 'FloorPlan7_physics',
                        'range_x': (-1.0, -0.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 1.0),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_73 = {'name': 'FloorPlan7_physics',
                        'range_x': (-1.5, 1.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-0.5, 1.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_74 = {'name': 'FloorPlan7_physics',
                        'range_x': (-2.5, 2.5),  # left - right
                        'range_y': (0.0, 1.0),
                        'range_z': (1.5, 2.0),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_75 = {'name': 'FloorPlan7_physics',
                        'range_x': (0.5, 2.0),  # close - far
                        'range_y': (0.0, 1.0),
                        'range_z': (0.0, 2.0),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_76 = {'name': 'FloorPlan7_physics',
                        'range_x': (-1.5, 0.5),  # close - far
                        'range_y': (0.0, 1.0),
                        'range_z': (0.0, 1.5),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_77 = {'name': 'FloorPlan7_physics',
                        'range_x': (-3.0, -2.5),  # close - far
                        'range_y': (0.0, 1.0),
                        'range_z': (0.0, 2.0),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_78 = {'name': 'FloorPlan7_physics',
                        'range_x': (-1.0, 0.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-0.5, 0.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan8_physics
    scene_physics_80 = {'name': 'FloorPlan8_physics',
                        'range_x': (-0.5, 0.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.0, 0.5),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_81 = {'name': 'FloorPlan8_physics',
                        'range_x': (-1.0, -0.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.0, 2.0),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_82 = {'name': 'FloorPlan8_physics',
                        'range_x': (-2.5, -1.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-1.0, -0.5),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_83 = {'name': 'FloorPlan8_physics',
                        'range_x': (-2.5, 0.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (-0.5, -0.3),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_84 = {'name': 'FloorPlan8_physics',
                        'range_x': (-0.5, 0.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (-0.5, 1.0),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan9_physics
    scene_physics_90 = {'name': 'FloorPlan9_physics',
                        'range_x': (-1.0, 2.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 0.25),  # close - far
                        'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_91 = {'name': 'FloorPlan9_physics',
                        'range_x': (0.5, 2.5),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 0.75),  # left - right
                        'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_92 = {'name': 'FloorPlan9_physics',
                        'range_x': (-1.0, 0.0),  # close - far
                        'range_y': (0.5, 1.0),
                        'range_z': (0.0, 0.75),  # left - right
                        'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_93 = {'name': 'FloorPlan9_physics',
                        'range_x': (-1.0, 2.5),  # left - right
                        'range_y': (0.5, 1.0),
                        'range_z': (0.75, 0.9),  # close - far
                        'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan10_physics
    scene_physics_100 = {'name': 'FloorPlan10_physics',
                         'range_x': (-1.5, 0.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.75, -1.25),  # left - right
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_101 = {'name': 'FloorPlan10_physics',
                         'range_x': (-4.0, -2.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.75, -1.0),  # left - right
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan11_physics
    scene_physics_110 = {'name': 'FloorPlan11_physics',
                         'range_x': (-2.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, -0.8),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_111 = {'name': 'FloorPlan11_physics',
                         'range_x': (0.0, 2.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 0.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_112 = {'name': 'FloorPlan11_physics',
                         'range_x': (-2.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-0.25, -0.25),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_113 = {'name': 'FloorPlan11_physics',
                         'range_x': (-2.5, 2.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 0.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan12_physics
    scene_physics_120 = {'name': 'FloorPlan12_physics',
                         'range_x': (0.75, 0.75),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-0.5, 2.5),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_121 = {'name': 'FloorPlan12_physics',
                         'range_x': (0.0, 0.75),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 0.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_122 = {'name': 'FloorPlan12_physics',
                         'range_x': (0.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (1.5, 2.0),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_123 = {'name': 'FloorPlan12_physics',
                         'range_x': (0.0, 0.25),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-0.5, 2.5),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan13_physics
    scene_physics_130 = {'name': 'FloorPlan13_physics',
                         'range_x': (-2.0, -0.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (2.5, 6.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_131 = {'name': 'FloorPlan13_physics',
                         'range_x': (-3.0, -2.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (2.5, 6.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_132 = {'name': 'FloorPlan13_physics',
                         'range_x': (-3.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (4.5, 5.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_133 = {'name': 'FloorPlan13_physics',
                         'range_x': (-2.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (2.25, 2.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan14_physics
    scene_physics_140 = {'name': 'FloorPlan14_physics',
                         'range_x': (-1.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-0.5, 0.0),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_141 = {'name': 'FloorPlan14_physics',
                         'range_x': (-1.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 0.5),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_142 = {'name': 'FloorPlan14_physics',
                         'range_x': (-0.5, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (2.5, 3.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_143 = {'name': 'FloorPlan14_physics',
                         'range_x': (-1.5, -1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 3.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan15_physics
    scene_physics_150 = {'name': 'FloorPlan15_physics',
                         'range_x': (-2.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (1.5, 2.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_151 = {'name': 'FloorPlan15_physics',
                         'range_x': (-2.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (1.5, 2.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan16_physics
    scene_physics_160 = {'name': 'FloorPlan16_physics',
                         'range_x': (1.25, 1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 1.5),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan17_physics
    scene_physics_170 = {'name': 'FloorPlan17_physics',
                         'range_x': (-0.5, 0.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 1.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_171 = {'name': 'FloorPlan17_physics',
                         'range_x': (-0.5, 0.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (1.0, 1.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_172 = {'name': 'FloorPlan17_physics',
                         'range_x': (-0.5, 0.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (0.5, 1.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_173 = {'name': 'FloorPlan17_physics',
                         'range_x': (-0.5, 0.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (2.0, 2.5),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_174 = {'name': 'FloorPlan17_physics',
                         'range_x': (-0.5, 0.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (3.0, 3.5),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan18_physics
    scene_physics_180 = {'name': 'FloorPlan18_physics',
                         'range_x': (-2.0, -0.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.0, 6.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_181 = {'name': 'FloorPlan18_physics',
                         'range_x': (-2.0, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (3.0, 4.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_182 = {'name': 'FloorPlan18_physics',
                         'range_x': (-2.0, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (4.0, 4.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan19_physics
    scene_physics_190 = {'name': 'FloorPlan19_physics',
                         'range_x': (-1.5, -1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-3.0, 0.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_191 = {'name': 'FloorPlan19_physics',
                         'range_x': (-2.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, 0.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_192 = {'name': 'FloorPlan19_physics',
                         'range_x': (-2.0, -1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, -1.5),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_193 = {'name': 'FloorPlan19_physics',
                         'range_x': (-2.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, -2.0),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan20_physics
    scene_physics_201 = {'name': 'FloorPlan20_physics',
                         'range_x': (0.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, -0.5),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_202 = {'name': 'FloorPlan20_physics',
                         'range_x': (0.0, 1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (2.0, 2.5),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_203 = {'name': 'FloorPlan20_physics',
                         'range_x': (-1.25, -1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 1.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan21_physics
    scene_physics_210 = {'name': 'FloorPlan21_physics',
                         'range_x': (-1.0, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.8, -1.6),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_211 = {'name': 'FloorPlan21_physics',
                         'range_x': (-1.0, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-0.25, 0.0),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_212 = {'name': 'FloorPlan21_physics',
                         'range_x': (-1.0, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, -0.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_213 = {'name': 'FloorPlan21_physics',
                         'range_x': (-1.0, -0.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.5, 0.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan22_physics
    scene_physics_220 = {'name': 'FloorPlan22_physics',
                         'range_x': (-1.5, -1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 1.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_221 = {'name': 'FloorPlan22_physics',
                         'range_x': (-2.0, -1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 1.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan23_physics
    scene_physics_230 = {'name': 'FloorPlan23_physics',
                         'range_x': (-1.5, -1.25),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, 0.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_231 = {'name': 'FloorPlan23_physics',
                         'range_x': (-4.5, -4.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, 0.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_232 = {'name': 'FloorPlan23_physics',
                         'range_x': (-1.7, -1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, 0.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan24_physics
    scene_physics_240 = {'name': 'FloorPlan24_physics',
                         'range_x': (-0.75, -0.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.5, 3.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_241 = {'name': 'FloorPlan24_physics',
                         'range_x': (-0.5, -0.25),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.5, 3.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_242 = {'name': 'FloorPlan24_physics',
                         'range_x': (-1.5, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (2.0, 2.5),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan25_physics
    scene_physics_250 = {'name': 'FloorPlan25_physics',
                         'range_x': (-1.75, -1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.0, 2.0),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_251 = {'name': 'FloorPlan25_physics',
                         'range_x': (-2.0, -1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.0, 2.0),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_252 = {'name': 'FloorPlan25_physics',
                         'range_x': (-2.0, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (1.5, 2.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan26_physics
    scene_physics_260 = {'name': 'FloorPlan26_physics',
                         'range_x': (-2.0, -1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.0, 2.5),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_261 = {'name': 'FloorPlan26_physics',
                         'range_x': (-1.5, -0.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (1.0, 2.5),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan27_physics
    scene_physics_271 = {'name': 'FloorPlan27_physics',
                         'range_x': (1.0, 1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 1.5),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_272 = {'name': 'FloorPlan27_physics',
                         'range_x': (0.5, 1.0),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 1.5),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_273 = {'name': 'FloorPlan27_physics',
                         'range_x': (0.5, 1.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (0.0, 1.5),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan28_physics
    scene_physics_280 = {'name': 'FloorPlan28_physics',
                         'range_x': (-2.5, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.0, -1.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_281 = {'name': 'FloorPlan28_physics',
                         'range_x': (-2.5, -1.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, -2.0),  # close - far
                         'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_282 = {'name': 'FloorPlan28_physics',
                         'range_x': (-2.0, -1.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-2.5, -1.5),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan29_physics
    scene_physics_290 = {'name': 'FloorPlan29_physics',
                         'range_x': (1.25, 1.25),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 0.5),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_291 = {'name': 'FloorPlan29_physics',
                         'range_x': (-0.5, 0.0),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (-0.2, 0.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan30_physics
    scene_physics_300 = {'name': 'FloorPlan30_physics',
                         'range_x': (2.0, 2.25),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 0.5),  # left - right
                         'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_301 = {'name': 'FloorPlan30_physics',
                         'range_x': (2.25, 2.5),  # close - far
                         'range_y': (0.5, 1.0),
                         'range_z': (-1.0, 0.5),  # left - right
                         'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_302 = {'name': 'FloorPlan30_physics',
                         'range_x': (0.0, 2.5),  # left - right
                         'range_y': (0.5, 1.0),
                         'range_z': (0.75, 1.0),  # close - far
                         'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan201_physics
    scene_physics_2010 = {'name': 'FloorPlan201_physics',
                          'range_x': (-4.5, -4.25),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (0.0, 5.0),  # left - right
                          'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2011 = {'name': 'FloorPlan201_physics',
                          'range_x': (-0.7, -0.5),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (1.0, 2.5),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2012 = {'name': 'FloorPlan201_physics',
                          'range_x': (-0.75, -0.5),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (4.0, 5.0),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2013 = {'name': 'FloorPlan201_physics',
                          'range_x': (-4.0, -0.5),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (2.5, 2.5),  # close - far
                          'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}
    
    scene_physics_2014 = {'name': 'FloorPlan201_physics',
                          'range_x': (-4.0, -0.5),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (2.5, 2.5),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_2015 = {'name': 'FloorPlan201_physics',
                          'range_x': (-4.5, -0.5),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (4.25, 4.5),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan202_physics
    scene_physics_2020 = {'name': 'FloorPlan202_physics',
                          'range_x': (-0.25, 0.0),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (1.5, 3.0),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan203_physics
    scene_physics_2030 = {'name': 'FloorPlan203_physics',
                          'range_x': (-3.0, -2.5),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (1.5, 5.0),  # left - right
                          'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2031 = {'name': 'FloorPlan203_physics',
                          'range_x': (-3.25, -2.75),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (0.0, 5.5),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan204_physics
    scene_physics_2040 = {'name': 'FloorPlan204_physics',
                          'range_x': (0.25, 0.5),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (3.0, 4.5),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2041 = {'name': 'FloorPlan204_physics',
                          'range_x': (-2.0, 0.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (2.5, 3.0),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan205_physics
    scene_physics_2050 = {'name': 'FloorPlan205_physics',
                          'range_x': (-3.5, -3.0),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (0.5, 2.5),  # left - right
                          'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2051 = {'name': 'FloorPlan205_physics',
                          'range_x': (-4.0, -2.5),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (2.0, 2.5),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan206_physics
    scene_physics_2060 = {'name': 'FloorPlan206_physics',
                          'range_x': (1.75, 2.0),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (-1.0, 0.5),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2061 = {'name': 'FloorPlan206_physics',
                          'range_x': (1.75, 2.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (-1.0, -0.5),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_2062 = {'name': 'FloorPlan206_physics',
                          'range_x': (-1.0, -0.75),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (-1.0, 0.5),  # left - right
                          'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan207_physics
    scene_physics_2070 = {'name': 'FloorPlan207_physics',
                          'range_x': (0.5, 1.0),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (-1.5, 1.5),  # left - right
                          'rotation': (0, 270, 0), 'axis': CameraArrayAlongAxis.Z}

    scene_physics_2071 = {'name': 'FloorPlan207_physics',
                          'range_x': (-1.0, 1.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (-1.5, -1.25),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan208_physics
    scene_physics_2080 = {'name': 'FloorPlan208_physics',
                          'range_x': (-1.0, 1.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (-1.75, -1.5),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan209_physics
    scene_physics_2090 = {'name': 'FloorPlan209_physics',
                          'range_x': (-5.0, -2.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (-0.75, -0.5),  # close - far
                          'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_2091 = {'name': 'FloorPlan209_physics',
                          'range_x': (-5.0, -2.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (-4.75, -4.5),  # close - far
                          'rotation': (0, 0, 0), 'axis': CameraArrayAlongAxis.X}

    scene_physics_2092 = {'name': 'FloorPlan209_physics',
                          'range_x': (-5.0, -2.0),  # left - right
                          'range_y': (0.5, 1.0),
                          'range_z': (-3.75, -3.5),  # close - far
                          'rotation': (0, 180, 0), 'axis': CameraArrayAlongAxis.X}

    # 'FloorPlan210_physics
    scene_physics_2100 = {'name': 'FloorPlan210_physics',
                          'range_x': (-6.0, -5.8),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (1.0, 3.5),  # left - right
                          'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    # 'FloorPlan211_physics
    scene_physics_2110 = {'name': 'FloorPlan211_physics',
                          'range_x': (-0.5, 0.0),  # close - far
                          'range_y': (0.5, 1.0),
                          'range_z': (-0.5, 1.5),  # left - right
                          'rotation': (0, 90, 0), 'axis': CameraArrayAlongAxis.Z}

    scenes = [scene_physics_11, scene_physics_12, scene_physics_13, scene_physics_14,
              scene_physics_21, scene_physics_22, scene_physics_23, scene_physics_24,
              scene_physics_31, scene_physics_32, scene_physics_33, scene_physics_34,
              scene_physics_40, scene_physics_41, scene_physics_42, scene_physics_43, scene_physics_44,
              scene_physics_51, scene_physics_52, scene_physics_53, scene_physics_54,
              scene_physics_60, scene_physics_61, scene_physics_62, scene_physics_63, scene_physics_64,
              scene_physics_70, scene_physics_71, scene_physics_72, scene_physics_73, scene_physics_74,
              scene_physics_75, scene_physics_76, scene_physics_77, scene_physics_78,
              scene_physics_80, scene_physics_81, scene_physics_82, scene_physics_83, scene_physics_84,
              scene_physics_90, scene_physics_91, scene_physics_92, scene_physics_93,
              scene_physics_100, scene_physics_101,
              scene_physics_110, scene_physics_111, scene_physics_112, scene_physics_113,
              scene_physics_120, scene_physics_121, scene_physics_122, scene_physics_123,
              scene_physics_130, scene_physics_131, scene_physics_132, scene_physics_133,
              scene_physics_140, scene_physics_141, scene_physics_142, scene_physics_143,
              scene_physics_150, scene_physics_151, scene_physics_160,
              scene_physics_170, scene_physics_171, scene_physics_172, scene_physics_173, scene_physics_174,
              scene_physics_180, scene_physics_181, scene_physics_182,
              scene_physics_190, scene_physics_191, scene_physics_192, scene_physics_193,
              scene_physics_201, scene_physics_202, scene_physics_203,
              scene_physics_210, scene_physics_211, scene_physics_212, scene_physics_213,
              scene_physics_220, scene_physics_221, scene_physics_230, scene_physics_231, scene_physics_232,
              scene_physics_240, scene_physics_241, scene_physics_242,
              scene_physics_250, scene_physics_251, scene_physics_252,
              scene_physics_260, scene_physics_261, scene_physics_271, scene_physics_272, scene_physics_273,
              scene_physics_280, scene_physics_281, scene_physics_282,
              scene_physics_290, scene_physics_291, scene_physics_300, scene_physics_301, scene_physics_302,
              scene_physics_2010, scene_physics_2011, scene_physics_2012, scene_physics_2013, scene_physics_2014,
              scene_physics_2015, scene_physics_2020, scene_physics_2030, scene_physics_2031,
              scene_physics_2040, scene_physics_2041, scene_physics_2050, scene_physics_2051,
              scene_physics_2060, scene_physics_2061, scene_physics_2062,
              scene_physics_2070, scene_physics_2071, scene_physics_2080,
              scene_physics_2090, scene_physics_2091, scene_physics_2092, scene_physics_2100, scene_physics_2110]

    return scenes


aithor_handling('/scratch/antruong/workspace/test/')
