""" QtImageViewer.py: PyQt image viewer widget for a QPixmap in a QGraphicsView scene with mouse zooming and panning.
"""

import os
import cv2
import pickle
import numpy as np

from aithor_utils import get_list_scenes, get_ai2thor_controller


def aithor_handling(out_path='/scratch/antruong/workspace/test/', manual_check=False):
    controller = get_ai2thor_controller()
    scenes = get_list_scenes()

    for _, s in enumerate(scenes):
        write_frame(out_path, s, controller, manual_check)


def write_frame(out_path, scene, controller, manual_check):
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
                    if not in_good_range(i, ry, k, max_z, max_x, min_z, min_x):
                        continue

                    event = controller.step(action='TeleportFull', x=i, y=j, z=k, rotation=dict(x=0, y=ry, z=0),
                                            horizon=0.0)

                    x, y, z = get_location(event)

                    if abs(i - x) + abs(j - y) + abs(k - k) > 0.05:
                        continue

                    print(i, j, k, x, y, z, ' --- ', ry, scene)

                    image = event.frame
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    depth = event.depth_frame

                    if manual_check:
                        cv2.imshow('Image', image)
                        if cv2.waitKey() == ord('n'):  # press n to reject the image
                            continue
                    else:
                        norm_depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
                        edge = (cv2.Canny(norm_depth, threshold1=0, threshold2=50) > 0).astype(float)

                        if np.sum(edge) < 10:
                            continue

                    extrinsic = compute_extrinsic(event, y)

                    if ry == 0 or ry == 180:
                        if z not in scene_dict[ry].keys():
                            scene_dict[ry][z] = dict()
                            scene_dict[ry][z]['count'] = 0

                        if y not in scene_dict[ry][z].keys():
                            scene_dict[ry][z][y] = dict()

                        scene_dict[ry][z]['count'] += 1
                        scene_dict[ry][z][y][x] = {'x': x, 'y': y, 'z': z, 'depth': depth, 'image': image,
                                                   'extrinsic': extrinsic}
                    else:
                        if x not in scene_dict[ry].keys():
                            scene_dict[ry][x] = dict()
                            scene_dict[ry][x]['count'] = 0

                        if y not in scene_dict[ry][x].keys():
                            scene_dict[ry][x][y] = dict()

                        scene_dict[ry][x]['count'] += 1
                        scene_dict[ry][x][y][z] = {'x': x, 'y': y, 'z': z, 'depth': depth, 'image': image,
                                                   'extrinsic': extrinsic}

    save_to_file(scene_dict, out_path, scene)


def compute_extrinsic(event, y):
    r1 = event.metadata['agent']['r1']
    r2 = event.metadata['agent']['r2']
    r3 = event.metadata['agent']['r3']
    t = event.metadata['agent']['t']

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

    return extrinsic


def get_location(event):
    position = event.metadata['agent']['position']
    x, y, z = position['x'], position['y'], position['z']

    return x, y, z


def in_good_range(idx, ry, k, max_z, max_x, min_z, min_x):
    if ry == 0 and k + 1.5 > max_z:
        return False

    if ry == 90 and idx + 1.5 > max_x:
        return False

    if ry == 180 and k - 1.5 < min_z:
        return False

    if ry == 270 and idx - 1.5 < min_x:
        return False

    if (ry == 0 or ry == 180) and (idx > max_x - 1.5 or idx < min_x + 1.5):
        return False

    if (ry == 90 or ry == 270) and (k > max_z - 1.5 or idx < min_z + 1.5):
        return False

    return True


def save_to_file(scene_dict, out_path, scene):
    # TO REMOVE REDUNDANT CAMERA ALONG AXIS X
    for ry in (0, 90, 180, 270):
        to_remove = []
        for i in scene_dict[ry].keys():
            if scene_dict[ry][i]['count'] <= 1:
                to_remove.append(i)

        for k in to_remove:
            scene_dict[ry].pop(k, None)

    for ry in (0, 90, 180, 270):
        for x in scene_dict[ry].keys():
            pickle.dump({'rotation': ry, 'data': scene_dict[ry][x], 'name': scene},
                        open(os.path.join(out_path, scene + '_' + str(ry) + '_' + str(x)), 'wb'))


output_folder = '/scratch/antruong/workspace/test/'
aithor_handling(output_folder)
