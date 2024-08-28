# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os

import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)

iamges_path = "./collect_data"  # 手眼标定采集的标定版图片所在路径
arm_pose_file = "./collect_data/poses.txt"  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    return R, t


def camera_calibrate(iamges_path):
    print("++++++++++开始相机标定++++++++++++++")
    # 角点的个数以及棋盘格间距
    XX = 9  # 标定板的中长度对应的角点的个数
    YY = 6  # 标定板的中宽度对应的角点的个数
    L = 0.035  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    for i in range(0, 20):  # 标定好的图片在iamges_path路径下，从0.jpg到x.jpg   一般采集20张左右就够，实际情况可修改

        image = f"{iamges_path}/{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):

            img = cv2.imread(image)
            print(f"图像大小： {img.shape}")
            # h_init, width_init = img.shape[:2]
            # img = cv2.resize(src=img, dsize=(width_init // 2, h_init // 2))
            # print(f"图像大小(resize)： {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            # print(corners)
            print(f"左上角点：{corners[0, 0]}")
            print(f"右下角点：{corners[-1, -1]}")

            # 绘制角点并显示图像
            cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
            cv2.imshow('Chessboard', img)

            cv2.waitKey(3000)  ## 停留1s, 观察找到的角点是否正确

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)

    # 标定得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # print("ret:", ret)
    print("内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs


def process_arm_pose(arm_pose_file):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        R, t = pose_to_homogeneous_matrix(pose=pose)
        R_arm.append(R)
        t_arm.append(t)
    return R_arm, t_arm


def hand_eye_calibrate():
    rvecs, tvecs = camera_calibrate(iamges_path=iamges_path)
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file)

    R, t = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    print("+++++++++++手眼标定完成+++++++++++++++")
    return R, t


if __name__ == "__main__":
    R, t = hand_eye_calibrate()

    print("旋转矩阵：")
    print(R)
    print("平移向量：")
    print(t)
