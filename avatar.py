import os
import json
import numpy as np
from math_lib import translate, scale, rotate

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

    def get_joint(self, controller_name):
        assert controller_name in self.controller_to_joints, '{} not in controller'.format(controller_name)
        return self.controller_to_joints[controller_name]

    def get_joints(self, controller_names):
        joints = []
        for cn in controller_names:
            joints.extend(self.controller_to_joints[cn])
        joints = sorted(list(set(joints)))
        return joints

class AvatarAnimationHierarchy:
    def __init__(self, json_path):
        data = json.load(open(json_path))['hierarchy']
        self.joint_short_to_full = {}
        self.joint_dict = {}
        for i in data:
            name = i['name']
            short_name = name.split('|')[-1]
            assert short_name not in self.joint_short_to_full
            self.joint_short_to_full[short_name] = name
            assert name not in self.joint_dict
            self.joint_dict[name] = i

    def get_name_by_short_name(self, short_name):
        return self.joint_short_to_full[short_name]

    def get_joint_by_short_name(self, short_name):
        name = self.joint_short_to_full[short_name]
        return self.joint_dict[name]

    def get_joint_by_name(self, name):
        return self.joint_dict[name]

    def get_transform(self, name):
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
            parentTotalMatrix = self.get_transform(joint['parents'])

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
        return mpTotalMatrix
