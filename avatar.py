import os
import json
import numpy as np
from math_lib import translate, scale, rotate, transform_to_translate_rotation_scale, translateM_to_vector, scalingM_to_vector, rotateM_to_vector

class AvatarController:
    def __init__(self, json_path):
        data = json.load(open(json_path))['controllers']
        self.controller_to_joints = {}
        self.controller_to_rotate_pivot = {}
        self.controller_to_scale_pivot = {}
        for i in data:
            if i['name'].rfind('Expression') != -1:
                continue
            short_name = i['name'].split('|')[-1]
            if len(i['joints']) != 0:
                assert short_name not in self.controller_to_joints.keys()
                self.controller_to_joints[short_name] = i['joints']
            assert short_name not in self.controller_to_rotate_pivot.keys()
            self.controller_to_rotate_pivot[short_name] = i['rotate_pivot']
            self.controller_to_scale_pivot[short_name] = i['scale_pivot']

    def get_rotate_pivot(self, controller_name):
        assert controller_name in self.controller_to_rotate_pivot, '{} not in controller'.format(controller_name)
        return self.controller_to_rotate_pivot[controller_name]

    def get_scale_pivot(self, controller_name):
        assert controller_name in self.controller_to_scale_pivot, '{} not in controller'.format(controller_name)
        return self.controller_to_scale_pivot[controller_name]

    def get_joint_short_names(self, controller_name):
        assert controller_name in self.controller_to_joints, '{} not in controller'.format(controller_name)
        return self.controller_to_joints[controller_name]

    def get_joints(self, controller_names):
        joints = []
        for cn in controller_names:
            joints.extend(self.controller_to_joints[cn])
        joints = sorted(list(set(joints)))
        return joints

    def get_controllers(self):
        return sorted(self.controller_to_joints.keys())

class AvatarAnimationHierarchy:
    def __init__(self, json_path):
        data = json.load(open(json_path))['hierarchy']
        self.joint_short_to_full = {}
        self.joint_dict = {}
        self.global_transform_dict = {}
        self.local_transform_dict = {}
        for i in data:
            name = i['name']
            short_name = name.split('|')[-1]
            assert short_name not in self.joint_short_to_full
            self.joint_short_to_full[short_name] = name
            assert name not in self.joint_dict
            self.joint_dict[name] = i

    def load_custom_shapes(self, json_path):
        data = json.load(open(json_path))['custom_shapes']
        for custom_shape in data:
            name = custom_shape['name']
            translationMat = translate(custom_shape['translation'])
            rotationMat = rotate(custom_shape['rotation'])
            scaleMat = scale(np.array(custom_shape['scaling']) + np.array([1,1,1]))
            customMat = np.dot(np.dot(translationMat, rotationMat), scaleMat)
            self.local_transform_dict[name] = customMat

    def export_custom_shapes(self, json_path):
        custom_shapes_dict = {"custom_shapes": []}
        for name in self.local_transform_dict.keys():
            custom_shape = {}
            custom_shape["name"] = name
            t, r, s = transform_to_translate_rotation_scale(self.local_transform_dict[name])
            custom_shape["rotation"] = list(rotateM_to_vector(r))
            custom_shape["scaling"] = list(scalingM_to_vector(s))
            custom_shape["translation"] = list(translateM_to_vector(t))
            custom_shapes_dict["custom_shapes"].append(custom_shape)
        json.dump(custom_shapes_dict, open(json_path, 'w'), indent = '    ')

    def get_name_by_short_name(self, short_name):
        return self.joint_short_to_full[short_name]

    def get_joint_by_short_name(self, short_name):
        name = self.joint_short_to_full[short_name]
        return self.joint_dict[name]

    def get_joint_by_name(self, name):
        return self.joint_dict[name]

    def get_global_transform(self, name):
        if name in self.global_transform_dict:
            return self.global_transform_dict[name]
        joint = self.get_joint_by_name(name)
        # use the joint's world info for parent
        if joint['parents'] == []:
            positionDiff = np.array(joint['world_position']) - np.array(joint['position'])
            rotationDiff = np.array(joint['world_rotation']) - np.array(joint['rotation']) - np.array(joint['orientation'])
            scaleDiff = np.array(joint['world_scaling']) - np.array(joint['scale']) + np.array([1,1,1])
            parentTranslationMatrix = translate(positionDiff)
            parentScaleMatrix = scale(scaleDiff)
            parentRotationMatrix = rotate(rotationDiff)
            parentTotalMatrix = np.dot(np.dot(parentTranslationMatrix, parentRotationMatrix), parentScaleMatrix)
        else:
            parentTotalMatrix = self.get_global_transform(joint['parents'])

        translationMat = translate(joint['position'])
        scaleMat = scale(joint['scale'])
        rotationMat = rotate(joint['rotation'])
        mpLocalMatrix = np.dot(np.dot(translationMat, rotationMat), scaleMat)

        # orientation
        mpOrientationMatrix = rotate(joint['orientation'])

        # total and bind
        mpTotalMatrix = np.dot(parentTotalMatrix, np.dot(mpLocalMatrix, mpOrientationMatrix))
        diff = np.linalg.norm(np.array([mpTotalMatrix[0][3], mpTotalMatrix[1][3], mpTotalMatrix[2][3]])-np.array(joint['world_position']))
        assert diff < 1e-9, 'sanity check fails: computed joint transform does not conform to world position, diff norm: {}'.format(diff)
        self.global_transform_dict[name] = mpTotalMatrix
        return mpTotalMatrix

    def get_local_transform(self, name):
        return self.local_transform_dict[name]

    def get_transform(self, name):
        return np.dot(self.get_global_transform(name), self.get_local_transform(name))

    def move_joint(self, name, dx, dy, dz):
        transform = self.get_transform(name)
        global_transform = self.get_global_transform(name)
        new_transform = transform.copy()
        new_transform[0][3] += dx
        new_transform[1][3] += dy
        new_transform[2][3] += dz
        # global_transform * new_local_transform = new_transform
        new_local_transform = np.dot(np.linalg.inv(global_transform), new_transform)
        self.local_transform_dict[name] = new_local_transform
