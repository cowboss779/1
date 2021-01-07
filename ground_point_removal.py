# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 21:38:46 2021

@author: cowboss779
"""
import os
import struct
import numpy as np
import mayavi.mlab as mlab
import open3d as o3d

def cal_dist_plane(P1, P2):
    '''計算平面歐幾里得距離'''
    x = P2[0] - P1[0]
    y = P2[1] - P1[1]
    d = x*x + y*y
    return d

def cal_slope(P1, P2):
    '''計算兩點斜率'''
    z = P2[2] - P1[2]
    d = cal_dist_plane(P1, P2)
    if d != 0:
        slope = z*z / d
    else:
        slope = 0
    return slope

def cal_sector_index(x, y, step):
    sector = int((np.arctan2(y,x) + np.pi) / step)
    return sector

def read_bin_file(path):
    '''讀取bin檔'''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for _, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def read_point_cloud(path, model):
    ''' 1: velodyne
        2: ouster
    '''
    ext = os.path.splitext(path)[1]
    if ext == '.bin':
        point_cloud = read_bin_file(path)
    elif ext == '.txt':
        point_cloud = np.loadtxt(path, comments='L')
    point_cloud = point_cloud*500
    if model == 'ouster':
        point_cloud = np.flipud(point_cloud)
    if point_cloud.shape[1] == 3:
        mark = np.full((len(point_cloud),1), -1)
        point_cloud = np.append(point_cloud, mark, axis=1)
    elif point_cloud.shape[1] > 3:
        point_cloud[:,3] = -1
    return point_cloud

def save_to_ply(pc, indx):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,0:3])
    o3d.io.write_point_cloud("H:/EMIS_dataset/train/frame%d_rm.ply"%(indx), pcd)

def draw_lidar(pc, g_pc, test_info=False, color=None, fig=None, 
               bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap='gnuplot', scale_factor=pts_scale, figure=fig)
    mlab.points3d(g_pc[:,0], g_pc[:,1], g_pc[:,2], color=(0,1,0), mode=pts_mode, colormap='gnuplot', scale_factor=pts_scale, figure=fig)

    if test_info:
        start = 30784
        end = 31784
        mlab.points3d(pc[start:end,0], pc[start:end,1], pc[start:end,2], color=(1,1,1), mode='sphere', scale_factor=0.2)
        mlab.text3d(pc[0,0], pc[0,1], pc[0,2], 'start: %f %f %f'%(pc[0,0], pc[0,1], pc[0,2]), scale=(0.1,0.1,0.1))
        mlab.text3d(pc[1,0], pc[1,1], pc[1,2], 'end: %f %f %f'%(pc[1,0], pc[1,1], pc[1,2]), scale=(0.1,0.1,0.1))

    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

    #draw axis
    mlab.text3d(2, 0, 0, 'x', scale=(0.25,0.25,0.25))
    mlab.text3d(0, 2, 0, 'y', scale=(0.25,0.25,0.25))
    mlab.text3d(0, 0, 2, 'z', scale=(0.25,0.25,0.25))
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def ground_point_remove(point_data, scan_index):
    disp_far = 1000000 # 1000000 = 2公尺 = (2*500)^2
    disp_near = 625  # 625 = 5公分 = (0.05*500)^2
    s_parameter = 100
    Tadj = 0.9
    H_thr = 200 # 40cm
    Lidar_Height = 865
    zero = np.array([0, 0, 0])

    for _, e in enumerate(scan_index):
        previous = -1
        scan_line = int(e[1]-e[0]+1)
        CTN = False

        for j in range(scan_line):
            current = j + int(e[0])

            if previous < 0:
                if np.abs(point_data[current,2]+Lidar_Height) <= H_thr:
                    previous = current
                    CTN = True
                else:
                    #previous = current
                    point_data[current,3] = False
                continue

            disp = cal_dist_plane(point_data[current], point_data[previous])
            slope_thr = 0.3

            if disp >= disp_far:
                slope_thr += (1 - (disp / disp_far)) / s_parameter
                if slope_thr < 0:
                    slope_thr = 0

            if disp <= disp_near:
                slope_thr = slope_thr + Tadj*(1-(disp/disp_near))

            if cal_slope(point_data[current], point_data[previous]) <= slope_thr: #(slope_thr*slope_thr)
                if cal_dist_plane(point_data[current], zero) > cal_dist_plane(point_data[previous], zero):
                    if CTN:
                        point_data[current,3] = True
                        point_data[previous,3] = True
                    else:
                        CTN = True
                    previous = current
            if current - previous > 1:
                point_data[current,3] = False
                CTN = False
    return point_data

if __name__=='__main__':
    kitti_path = ['data/2011_09_26_0060_0000000000.txt',
                  'data/2011_09_28_0016_0000000000.txt',
                  'data/kitti360/kitti360_0000000000.txt',
                  'data/2011_09_28_0037_0000000000.bin',
                  'data/2011_09_28_0016_0000000000.bin',
                  'C:/Users/cowboss779/Desktop/frame0.txt']
    
    txt_path = 'H:/EMIS_dataset/OusterOS1_20201216_xyz/frame%d.txt'
    lidar_model = [['velodyne', 2048, 64],['ouster', 2048, 64]]
    which_model = lidar_model[1][0]
    
    for file_indx in range(50):
        sorted_pc = read_point_cloud(txt_path%(file_indx), which_model)
    
        if which_model == 'velodyne':
            SECTOR_SIZE = lidar_model[0][1]
            SEGMENT_STEP = 2*np.pi/SECTOR_SIZE # 每個扇區的角度
            mapped_pc = np.full((SECTOR_SIZE,64), -1)
            for idx, e in enumerate(sorted_pc):
                S = cal_sector_index(e[0], e[1], SEGMENT_STEP)
                if S == SECTOR_SIZE:
                    S = 0
                bin_position = np.asarray(np.where(mapped_pc[S] == -1))
                mapped_pc[S, bin_position.shape[1]-1] = idx
    
            for i in range(SECTOR_SIZE):
                one_line = mapped_pc[i, np.where(mapped_pc[i,:] > -1)]
                one_line = np.reshape(one_line, (-1,1))
                pc_one = sorted_pc[one_line[:,0]]
                ground_point_remove(pc_one, np.array([[0, len(pc_one)-1]]))
                for j, e in enumerate(one_line):
                    sorted_pc[e,3] = pc_one[j,3]
        else:
            SCAN_LINES = lidar_model[1][1]
            LAYERS = lidar_model[1][2]
            #pc_flipud = sorted_pc
            for i in range(SCAN_LINES):
                temp_64 = sorted_pc[i*LAYERS:i*LAYERS+LAYERS]
                #temp_idx = np.asarray(np.where(temp_64[:,3] == -2))
                temp_idx = np.where((temp_64[:,0] != 0) & (temp_64[:,1] != 0) & (temp_64[:,2] != 0))
                temp_idx = np.reshape(temp_idx,(-1,1))
                #temp_2 = temp_64[np.where(temp_64[:,3] == -2)]
                temp_pc = temp_64[np.where((temp_64[:,0] != 0) & (temp_64[:,1] != 0) & (temp_64[:,2] != 0))]
                ground_point_remove(temp_pc, np.array([[0, len(temp_pc)-1]]))
                for j in range(len(temp_pc)):
                    sorted_pc[i*LAYERS+temp_idx[j],3] = temp_pc[j,3]
    
        not_gp = sorted_pc[np.where(sorted_pc[:,3] == 0)]
        gp = sorted_pc[np.where(sorted_pc[:,3] == 1)]
        not_gp[:,0:3] /= 500
        #test_gp = not_GP[np.where(abs(not_GP[:,0]) < 2.5)]
        #test_gp = test_gp[np.where(abs(test_gp[:,1]) < 2.5)]
        gp[:,0:3] /= 500
        save_to_ply(not_gp, file_indx)
        print('%d done'%file_indx)
        print('non ground point: %d'%(len(not_gp)))
        print('ground point: %d'%(len(gp)))
        print('')
#        fig = draw_lidar(not_gp, gp)
#        mlab.show()
    