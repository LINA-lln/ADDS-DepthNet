import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import scipy.io as sio
from PIL import  Image
import open3d as o3d


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):

        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
      ##  self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = calibs['Tr'] ##
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.Tr = np.row_stack((self.V2C,np.array([0,0,0,1])))
        self.P2 = self.P.dot(self.Tr)
        self.P3 = np.row_stack((self.P, np.array([0,0,0,1])))
        self.inv_P3 = np.linalg.inv(self.P3)  #4*4

        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
      ##  self.R0 = calibs['R0_rect']
      ##  self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        ## pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_rect) ##self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        # a = uv_depth[:,2]
        # uv_depth[:,0] = uv_depth[:,0]*uv_depth[:,2]
        # uv_depth[:,1] = uv_depth[:,1]*uv_depth[:,2]
        # uv_depth2 = np.row_stack((uv_depth.T,np.ones(uv_depth.shape[0])))
        # velo_pred = self.inv_P3.dot(uv_depth2)
        # velo_pred = velo_pred.T
        # velo_pred = velo_pred[:,0:3]
        pts_3d_rect = self.project_image_to_rect(uv_depth)


        return self.project_rect_to_velo(pts_3d_rect)  #2 lines right


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def resize_and_crop(pilimg,  height, weight,scale=0.5,):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    diffH = newH - height
    diffW = newW - weight

    img = pilimg.resize((newW, newH))
    img = img.crop((diffW // 2, diffH // 2, newW- diffW // 2, newH - diffH // 2))
    return np.array(img, dtype=np.float32)

def to_cropped_imgs(ids, dir, suffix, scale, height, weight):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # plt.imsave(('val_img.png' ), np.asarray(Image.open(dir + id + suffix)).astype(np.uint8))
        im = resize_and_crop(Image.open(dir + id + suffix), height, weight, scale=scale)
        yield im   #get_square(im, pos)

##############################
def to_cropped_depths(depth, scale, height, weight):
    """From a list of tuples, returns the correct cropped img"""

    ndarray_convert_img = Image.fromarray(depth)
    im = resize_and_crop(ndarray_convert_img, height, weight, scale=scale)
    return im   #get_square(im, pos)


def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T

    cloud = calib.project_image_to_velo(points)

    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    #add
    cloud = cloud[valid]#add last
    valid2 = (cloud[:,2]<0.3) | (cloud[:,0]>10)

    return cloud[valid2]


path = '/home/liu/my_mnt/nyudv2_baseline/1215.mat'

dep = sio.loadmat(path)["depth"]
depth = to_cropped_depths(dep,0.6,256,320)
depth = depth /1000.0

print(depth)


path1 = '/home/liu/my_mnt/论文-depth/1204/val_0061_spa.png'
sparse_png = plt.imread(path1)
sparse_png = np.transpose(sparse_png,(2,0,1))

mask = np.zeros_like(depth)
mask[sparse_png[1]!=0]=1

sparse = depth*mask

calib_file = '/home/liu/my_mnt/kitti-object/pseudo_lidar/KITTI/object/test_recognition/calib/00/calib.txt'
calib = Calibration(calib_file)

lidar = project_depth_to_points(calib, sparse, 100)
lidar = lidar.astype(np.float32)

# open3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar)
# opt = o3d.visualization.get_render_option()
# o3d.visualization.background_color = np.asarray([0, 0, 0])
# pcd.paint_uniform_color([0, 0, 0])
# o3d.visualization.setBackgroundColor(0, 0, 0)
# opt = o3d.visualization.Visualizer().get_render_option()
# opt.background_color = np.asarray([0, 0, 0])

o3d.visualization.draw_geometries([pcd])

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(lidar)
#
# ctr = vis.get_view_control()
#
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])
# vis.run()



print(sparse_png)



