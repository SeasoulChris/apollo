#!/usr/bin/env python

import collections
import copy
import math
import os

from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np

from fueling.perception.YOLOv3.utils.object_utils import Calibration


def read_txt(path):
    """
    Read a txt file and parse it into lines.
    """
    with open(path) as handle:
        lines = handle.readlines()
    if len(lines) > 0 and lines[-1] == '\n':
        return lines[:-1]
    else:
        return lines

def read_camera_params(path):
    return Calibration(path)

def read_kitti_label(path, class_name_id_map):
    """
    [0:truncated, 1:occluded, 2:alpha, 3:x0, 4:y0, 5:x1, 6:y1, 
     7:h, 8:w, 9:l, 10:X, 11:Y, 12:Z, 13:rotation_y]
    """
    lines = read_txt(path)
    objs = collections.defaultdict(list)
    for obj in lines:
        obj_name = obj.split()[0]
        objs[obj_name].append(Label_Object(obj, class_name_id_map))
    return objs

class Camera(object):   
    def __init__(self, intrinsic, rx=0, ry=0, rz=0, tx=0, ty=0, tz=0):
        assert intrinsic.shape==(3, 3), "Shape of intrinsic must be (3, 3)"
        self.R = self.rotation_matrix(rx, ry, rz)
        self.T = self.translation_vector(tx, ty, tz)
        self.intrinsic = intrinsic
        self.fx, self.fy, self.cx, self.cy = \
            intrinsic[0, 0], intrinsic[1,1], intrinsic[0, 2], intrinsic[1, 2]
        
    def rotation_matrix(self, x=0, y=0, z=0):
        """
        Compute rotation matrix about x, y, z axis.
        x, y, z, float, angles to rotate in radians.
        Return: rotation matrix. shape=(3, 3)
        """
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(x), -np.sin(x)],
                       [0.0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(y), 0.0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0.0],
                       [np.sin(z), np.cos(z), 0.0],
                       [0.0, 0.0, 1.0]])
        return np.dot(np.dot(rz, ry), rx)
    
    def translation_vector(self, X, Y, Z):
        """
        Get the (3-by-1) translation vector. 
        X,Y,Z: the location of the object in the cam coor.
        Return: the translation to express points of obj coor in cam coor. shape = (3, 1)
                obj_point in obj-coor + translation = obj_point expressed in cam-coor.
        """
        return np.array([[X], [Y], [Z]])
    
    def set_R(self, R):
        """
        Set the 3-by-3 rotation matrix directly.
        """
        assert R.shape==(3, 3), "shape of R must be (3, 3)"
        self.R = R
        
    def set_T(self, T):
        """
        Set the 3-by-1 translation vector directly.
        """
        assert T.shape==(3, 1), "shape of T must be (3, 1)"
        self.T = T
    
    def convert_obj_to_camera(self, R, T, p):
        """
        Convert points by Rotation and Translation to camera coor.
        R: rotation matrix, (3, 3)
        T: translation, (3, 1)
        p: points, (3, n)
        return: shape (3, n)
        """
        return np.dot(R, p) + T #(3, n) + (3, 1) = (3, n)
    
    def project(self, points):
        """
        Project a point in cam coor to image plane.
        intrinsic: intrinsic matrix, shape=(3, 3)
        points: points in cam coor, shape=(n, 3) 
        return: xy coor in image, shape=(n, 2)
        """
        assert points.shape[1]==3, "Shape of points must be (n, 3)"
        
        XYZ = points.transpose() # (3, n)
        xyw = np.dot(self.intrinsic, XYZ) # (3, n)
        x = xyw[0:1, :] / xyw[2:3, :] #(1, n)
        y = xyw[1:2, :] / xyw[2:3, :] #(1, n)
        return np.concatenate([x,y], axis=0).transpose()  #(n, 2)

    def back_project(self, pix_coor):
        """
        Compute the vector backprojected from camera thru a pixel coordiante.
        pix_coor: np.array([[x], 
                            [y]]) of the pixel coordinate on the image plane. shape=(2, n)
        return: a direction vector of the back-projection. Have not been normalized. shape=(3, 1)
        """
        assert pix_coor.shape[0]==2, "Shape of pix_coor must be (2, n)"
        
        pix_coor = np.vstack([pix_coor, np.ones((1, pix_coor.shape[1]))])
        directions = np.dot(np.linalg.inv(self.intrinsic), pix_coor)

        # TDOD[KaWai]: uncomment below code to normalize the diretion vector.
        #directions = directions / np.linalg.norm(directions, ord=2) #(3, 1)
        return directions # (3,1)
    

class kitti_obj_cam_interaction(Camera):
    
    def __init__(self, calib):
        Camera.__init__(self, calib.P[:3, :3])
        self.calib = calib
        self.offset = np.dot(np.linalg.inv(calib.P[:3, :3]), calib.P[:, 3:4]) # (3, 1)

    def object_3d_points_in_obj_coor(self, obj):
        """
        From the h,w,l of the obj, compute the coordinates of the
            8 3d bbox points in the obj coor. 
        In the obj coor, x:forward, y:down, z:left.
            Origin at the center of the bottom of the object.
        Return: [front top left, top right, bottom right, bottom left,
                 rear top left, top right, bottom right, bottom left],
                shape = (8, 3)
        """
        h, w, l = obj.h, obj.w, obj.l
        return np.array([[l/2, -h, w/2], [l/2, -h, -w/2],
                         [l/2, 0, -w/2], [l/2, 0, w/2],
                         [-l/2, -h, w/2], [-l/2, -h, -w/2],
                         [-l/2, 0, -w/2], [-l/2, 0, w/2]])

    def rotate_aboutY(self, angle, p):
        """
        To rotate a point or vector about Y for angle radians.
        angle: float, radian. clockwise is +.
        p : The point/vector to rotate, shape=(3,1)
        Return: the rotated point/vector, shape = (3, 1)
        """
        return np.dot(self.rotation_matrix(y=angle), p)

    def transform_obj_to_camera(self, obj):
        """
        Transform a KITTI obj's 8 3d bbox points from obj coor to ref cam coor.
        obj: a KITTI Object
        return: 8 3D bbox points of the object in ref cam coor. shape = (8, 3)
        """
        angle = obj.ry
        X, Y, Z = obj.t

        R = self.rotation_matrix(y=angle) #(3, 3)
        T = self.translation_vector(X, Y, Z) #(3, 1)
        p = self.object_3d_points_in_obj_coor(obj) #(8, 3)

        p_camera = self.convert_obj_to_camera(R, T, p.transpose()) #(3, 8)
        return p_camera.transpose()

    def project_to_image(self, points, point_in_ref_cam=True):
        """
        Project a point in cam coor to image plane.
        intrinsic: the KITTI-style camera intrinsics, shape=(3, 4)
        points: points in cam coor, shape=(n, 3)
        point_in_ref_cam: bool, True if points are expressed in the reference
            camera coordiante.
        return: xy coor in image, shape=(2,n)
        """
        if point_in_ref_cam:
            points = points + self.offset.transpose()
        xy = self.project(points)  #(n, 2)
        return xy.transpose()  #(2, n)

    def move_directions_to_obj_coor(self, obj, points):
        """
        Translate a direction vector by T so that it starts at T, point to point+T.
        point: shape=(3, n)
        """
        assert points.shape[0] == 3, "Shape of points must be (3, n)"
        X, Y, Z = obj.t
        return points + self.translation_vector(X, Y, Z)
    
    def bbox_2d_direction(self, obj):
        """
        Compute back-projected 2d bbox ray.
        """
        ##=========== back projected 2D box direction ============
        #obj pixel center
        center_x = int(obj.xmax - obj.xmin) // 2 + int(obj.xmin)
        center_y = int(obj.ymax - obj.ymin) // 2 + int(obj.ymin)
        pix_coor = np.array([[center_x], [center_y]])
        #the back-projected 2D bbox direction
        bbox_direction = self.back_project(pix_coor)  *1 #(3, 1) 
        return bbox_direction
    
    def angle_btw_car_and_2d_bbox(self, obj):
        """
        Find angle btw bbox direction and obj direction on the camera x-z plane.
        """
        ##=========== back projected 2D box direction ============
        bbox_direction = self.bbox_2d_direction(obj)
        bbox_direction_on_xz = bbox_direction
        bbox_direction_on_xz[1] = 0

        #============= car direction ==============
        angle = obj.ry
        h, w, l = obj.h, obj.w, obj.l
        R = self.rotation_matrix(y=angle) #(3, 3)

        car_direction = np.dot(R, np.array([[1], [0], [0]]))
        car_direction_on_xz = car_direction
        car_direction_on_xz[1] = 0
        
        #============= compute angle ==============
        cos = np.dot(bbox_direction_on_xz.transpose(), car_direction_on_xz) /\
                 (np.linalg.norm(bbox_direction_on_xz, ord=2) *
                  np.linalg.norm(car_direction_on_xz, ord=2))
        side = car_direction_on_xz[0]*bbox_direction_on_xz[2] - \
               car_direction_on_xz[2]*bbox_direction_on_xz[0]
        if side > 0: #bbox_direction is on left of car_direction
            theta = 360 - math.degrees(math.acos(cos))
        else:
            theta = math.degrees(math.acos(cos))
        return theta
    
    def angle_btw_2d_bbox_and_x_axis(self, obj):
        """
        Find angle btw bbox direction and x-axis of cam coor on th x-z plane
        """
        bbox_direction = self.bbox_2d_direction(obj)
        bbox_direction_on_xz = bbox_direction
        bbox_direction_on_xz[1] = 0
        
        x_axis = np.array([[1],[0],[0]])
        cos = np.dot(bbox_direction_on_xz.transpose(), x_axis) / \
                (np.linalg.norm(bbox_direction_on_xz, ord=2))
        side = x_axis[0] * bbox_direction_on_xz[2] - x_axis[2] * bbox_direction_on_xz[0]
        if side > 0:  #bbox_direction is on left of x_axis
            alpha = math.degrees(math.acos(cos))
        else:
            alpha = 360 - math.degrees(math.acos(cos))
        return alpha
    
    def angle_btw_car_and_x_axis(self, obj):
        """
        Find the angle btw car direction and x-axis.
        """
        theta = self.angle_btw_car_and_2d_bbox(obj)
        alpha = self.angle_btw_2d_bbox_and_x_axis(obj)
        return (360 - (theta + alpha))
    
    def local_angle_to_car_yaw(self, local_angle, obj):
        """
        Compute the car yaw from local_angle(angle btw 2d bbox direction and car direction)
        Params:
        local_angle: the angle btw 2d bbox direction and car direction
        xmin, ymin, xmax, ymax: the 2d bbox
        """
        alpha = self.angle_btw_2d_bbox_and_x_axis(obj)
        return (360 - (local_angle + alpha))
    
    def bbox_from_local_angle_translation_dimension(self, obj, local_angle=None):
        """
        Compute the 3d bbox from local angle, translation and dimension and 2d bbox.
        Params:
        local_angle: the angle btw 2d bbox direction and car direction.
        obj: an Object insstance, must have h, w, l, X, Y, Z, xmin, ymin, xmax, ymax
        Return:
        The 8 points of the 3d bbox. (n, 3)
        """
        if local_angle and obj.ry == None:
            alpha = self.local_angle_to_car_yaw(local_angle, obj)
            obj.ry = math.radians(alpha)
        points = self.transform_obj_to_camera(obj)
        return points
    
    def find_2d_3d_correspondence(self, obj):
        """
        Find the correspondece between 2d edge and 3d points.
        Assuming 2d bbox center is close to obj center.
        Param:
            obj: must have xmin, yminb, xmax, ymax, h, w, l, ry
        Return:
            xmin_idx, xmax_idx, ymin_idx, ymax_idx
        """
        direction = self.bbox_2d_direction(obj)  #(3, 1)
        orig_translation = obj.t
        temp_translation = direction * 10
        obj.t = (temp_translation[0, 0], temp_translation[1, 0], temp_translation[2, 0])
        points_cam = self.bbox_from_local_angle_translation_dimension(obj) #(8, 3)
        points_image = self.project_to_image(points_cam, point_in_ref_cam=False) #(2, n)
        left = np.argmin(points_image[0, :])
        right = np.argmax(points_image[0, :])
        up = np.argmin(points_image[1, :])
        bottom = np.argmax(points_image[1, :])

        xmin, ymin, xmax, ymax = obj.box2d
        R = self.rotation_matrix(y=obj.ry)
        points = self.object_3d_points_in_obj_coor(obj)
        X0 = points[left]
        X1 = points[right]
        Y0 = points[up]
        Y1 = points[bottom]
        A = np.zeros((4, 4), dtype=np.float32)
        RX0 = np.dot(R, X0)
        RX1 = np.dot(R, X1)
        RY0 = np.dot(R, Y0)
        RY1 = np.dot(R, Y1)
        A[0, 0], A[0, 2], A[0, 3] = \
            self.fx, self.cx - xmin, self.fx * (RX0[0]) - xmin * (RX0[2]) + self.cx * (RX0[2])
        A[1, 0], A[1, 2], A[1, 3] = \
            self.fx, self.cx - xmax, self.fx * (RX1[0]) - xmax * (RX1[2]) + self.cx * (RX1[2])
        A[2, 1], A[2, 2], A[2, 3] = \
            self.fy, self.cy - ymin, self.fy * (RY0[1]) - ymin * (RY0[2]) + self.cy * (RY0[2])
        A[3, 1], A[3, 2], A[3, 3] = \
            self.fy, self.cy - ymax, self.fy * (RY1[1]) - ymax * (RY1[2]) + self.cy * (RY1[2])

        b = -A[:, 3]
        A = A[:4, :3]
        sol = np.linalg.lstsq(A, b)

        obj.t = (sol[0][0], sol[0][1], sol[0][2])
        points_cam = self.bbox_from_local_angle_translation_dimension(obj) #(8, 3)
        points_image = self.project_to_image(points_cam, point_in_ref_cam=False) #(2, n)
        left = np.argmin(points_image[0, :])
        right = np.argmax(points_image[0, :])
        up = np.argmin(points_image[1, :])
        bottom = np.argmax(points_image[1, :])

        obj.t = orig_translation
        return left, right, up, bottom

    def compute_translation(self, obj):
        """
        Compute translation of obj from 2d bbox, 3d bbox dimension and yaw.
        """
        if (np.all([obj.h, obj.w, obj.l, obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.ry]) == None):
            raise RuntimeError("obj need to have valid hwl, 2d bbox and ry")
            
        xmin, ymin, xmax, ymax = obj.box2d
        R = self.rotation_matrix(y=obj.ry)
        points = self.object_3d_points_in_obj_coor(obj) #(8, 3)
        left_idx, right_idx, up_idx, bottom_idx = self.find_2d_3d_correspondence(obj)
        
        # TODO(KaWai): uncomment below to go through all possible correspondence.
        #up_idx = [0, 1, 4, 5]
        #bottom_idx = [2, 3, 6, 7]
        #left_idx = [0, 1, 2, 3, 4, 5, 6, 7]
        #right_idx = [6, 7, 4, 5, 2, 3, 0, 1]
        residual = 1000000000
        ans = None
        for up in [up_idx]:
            Y0 = points[up]
            for bottom in [bottom_idx]:
                Y1 = points[bottom]
                for left, right in zip([left_idx], [right_idx]):
                    X0 = points[left]
                    X1 = points[right]
                    A = np.zeros((4, 4), dtype=np.float32)
                    RX0 = np.dot(R, X0)
                    RX1 = np.dot(R, X1)
                    RY0 = np.dot(R, Y0)
                    RY1 = np.dot(R, Y1)
                    A[0, 0], A[0, 2], A[0, 3] = self.fx, self.cx - xmin, \
                        self.fx * (RX0[0]) - xmin * (RX0[2]) + self.cx * (RX0[2])
                    A[1, 0], A[1, 2], A[1, 3] = self.fx, self.cx - xmax, \
                        self.fx * (RX1[0]) - xmax * (RX1[2]) + self.cx * (RX1[2])
                    A[2, 1], A[2, 2], A[2, 3] = self.fy, self.cy - ymin, \
                        self.fy * (RY0[1]) - ymin * (RY0[2]) + self.cy * (RY0[2])
                    A[3, 1], A[3, 2], A[3, 3] = self.fy, self.cy - ymax, \
                        self.fy * (RY1[1]) - ymax * (RY1[2]) + self.cy * (RY1[2])
                    b = -A[:, 3]
                    A = A[:4, :3]
                    sol = np.linalg.lstsq(A, b)
                    res = sol[1][0]
                    obj_temp = copy.deepcopy(obj)
                    obj_temp.t = (sol[0][0], sol[0][1], sol[0][2])
                    ry = self.angle_btw_car_and_x_axis(obj_temp)
                    if (res < residual):
                        ans = obj_temp.t
                        residual = res
        return ans
    
def draw_3d_box(img, points):
    """
    img: image in the form of np array.
    points: (n, 2), n image points to draw.
    """
    #img = Image.fromarray(img)
    width = 2
    draw = ImageDraw.Draw(img)
    draw.line([tuple(points[0, :]), tuple(points[1, :])], fill=(255,0,0), width=width)
    draw.line([tuple(points[1, :]), tuple(points[2, :])], fill=(255,0,0), width=width)
    draw.line([tuple(points[2, :]), tuple(points[3, :])], fill=(255,0,0), width=width)
    draw.line([tuple(points[3, :]), tuple(points[0, :])], fill=(255,0,0), width=width)
    
    draw.line([tuple(points[4, :]), tuple(points[5, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[5, :]), tuple(points[6, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[6, :]), tuple(points[7, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[7, :]), tuple(points[4, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[0, :]), tuple(points[4, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[1, :]), tuple(points[5, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[2, :]), tuple(points[6, :])], fill=(0,255,0), width=width)
    draw.line([tuple(points[3, :]), tuple(points[7, :])], fill=(0,255,0), width=width)
    
    # 8th point is location, 9th point is the forward direction,
    # 10th is the back-projected direction,
    # TODO(KaWai): uncomment code below to draw the forward direction.
    #draw.line([tuple(points[8, :]), tuple(points[9, :])], fill=(0,255,255), width=3)
    #draw.line([tuple(points[8, :]), tuple(points[10, :])], fill=(0,255,255), width=3)
    return np.array(img, dtype=np.uint8)
