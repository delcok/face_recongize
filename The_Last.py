#-*-coding:utf-8_*_
#作者      :71460
#创建时间  :2018/11/25 13:42
#文件      :The_Last.py
#IDE       :PyCharm

import dlib
import numpy as np
import cv2
import requests
from json import JSONDecoder

import datetime
import os
import shutil
#Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
#检测脸部特征
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
#调用摄像头
cap = cv2.VideoCapture(0)
#设置视频参数
cap.set(3,480)
#人脸截图计数器
cnt_ss = 0
current_face_dir = 0
path_make_dir = 'data/data_faces_from_camera/'

#识别人的年龄
age = 0
#识别人的性别
is_famle = ''
#face++的信息以及api接口
http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
#输入自己的key，和secret
key = '------------------------------------'
secret = '-----------------------------------'
#传输数据
attributes = 'gender,age,beauty,ethnicity,emotion,smiling,headpose'
data = {'api_key':key,'api_secret':secret,'return_attributes':attributes}
def pre_work():
    if os.path.isdir(path_make_dir):
        pass
    else:
        os.mkdir(path_make_dir)

    # 删除之前存的人脸数据文件夹
    folders_rd = os.listdir(path_make_dir)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_make_dir + folders_rd[i])


#进行删除数据
pre_work()
#人脸数目计数器
person_cnt = 0
#存储照片
save_flag = 1
while cap.isOpened():
    flag,img_rd = cap.read()
    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img_rd,cv2.COLOR_RGB2BGR)

    #人脸数
    faces = detector(img_gray,0)
    #字体
    font = cv2.FONT_HERSHEY_COMPLEX
    # 按下 'n' 新建存储人脸的文件夹
    if kk == ord('n'):
        person_cnt += 1
        current_face_dir = path_make_dir + "person_" + str(person_cnt)
        print('\n')
        for dirs in (os.listdir(path_make_dir)):
            if current_face_dir == path_make_dir + dirs:
                shutil.rmtree(current_face_dir)
                print("删除旧的文件夹:", current_face_dir)
        os.makedirs(current_face_dir)
        print("新建的人脸文件夹: ", current_face_dir)

        # 将人脸计数器清零
        cnt_ss = 0
    if len(faces) != 0:
        # 检测到人脸

        # 矩形框
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height / 2)
            ww = int(width / 2)

            # 设置颜色 / The color of rectangle of faces detected
            color_rectangle = (0, 0, 255)
            if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (d.left() - ww < 0) or (d.top() - hh < 0):
                cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (255, 255, 255)
                save_flag = 0
            else:
                color_rectangle = (0, 0, 255)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)
            im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)

            if save_flag:
                # 按下 's' 保存摄像头中的人脸到本地
                if kk == ord('s'):
                    cnt_ss += 1
                    for ii in range(height * 2):
                        for jj in range(width * 2):
                            im_blank[ii][jj] = img_rd[d.top() - hh + ii][d.left() - ww + jj]
                    cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", im_blank)
                    print("写入本地：", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    #进行人脸数据的反馈与接收
                    filepath = str(str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    files = {'image_file': open(filepath, 'rb')}
                    response = requests.post(http_url, data=data, files=files)
                    req_con = response.content.decode('utf-8')
                    req_dict = JSONDecoder().decode(req_con)
                    #赋值给age
                    age_4 = req_dict['faces']
                    age_3 = age_4[0]
                    age_2 = age_3['attributes']
                    age_1 = age_2['age']
                    age = age_1['value']
                    #赋值给is_famle
                    feamle5 = req_dict['faces']
                    feamle2 = feamle5[0]
                    feamle3 = feamle2['attributes']
                    feamle4 = feamle3['gender']
                    is_famle = feamle4['value']


    # 显示人脸数
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    #显示年龄
    cv2.putText(img_rd, "age: " + str(age), (20, 150), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    #显示男女
    cv2.putText(img_rd, "sex: " + str(is_famle), (20, 200), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    # 添加说明
    cv2.putText(img_rd, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "S: Recongize face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    # 按下 'q' 键退出
    if kk == ord('q'):
        break

    # 窗口显示
    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()