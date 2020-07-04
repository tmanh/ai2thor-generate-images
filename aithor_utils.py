import os
import pickle

import cv2

import ai2thor.controller

import numpy as np


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

                    image = event.frame
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    depth = event.depth_frame
                    print(depth.shape)

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


def get_ai2thor_controller():
    local_executable_path = '/scratch/antruong/workspace/ai2thor/unity/builds/thor-local-Linux64'
    controller = ai2thor.controller.Controller(local_executable_path=local_executable_path)
    controller.step(dict(action='ChangeResolution', x=600, y=600), raise_for_failure=True)
    return controller


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
