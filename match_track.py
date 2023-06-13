import cv2
import numpy as np
import time

# 初始化视频读取对象
cap = cv2.VideoCapture(0)

# 获取第一帧的图像
ret, frame = cap.read()

# 初始化ROI（手工选择的跟踪区域）
roi = cv2.selectROI("Frame", frame, False)

# 将选定的区域转换为灰度图像，并提取模板
template = cv2.cvtColor(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])], cv2.COLOR_BGR2GRAY)

# 初始化模板匹配方法
match_method = cv2.TM_CCOEFF_NORMED

# 初始化FPS计数器、模板更新计数器和计时器
fps = 0
update_count = 0
curr_time = time.time()

# 主循环，不断读取图像帧，并匹配模板
while True:
    # 获取当前帧
    ret, frame = cap.read()

    # 将当前帧转换成灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在当前帧上进行模板匹配，得到匹配结果
    res = cv2.matchTemplate(gray_frame, template, match_method)

    # 获取匹配结果的最大值和最小值，并对结果进行归一化处理
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        top_left = max_loc
    else:
        top_left = max_loc

    # 计算ROI的坐标
    bottom_right = (top_left[0] + roi[2], top_left[1] + roi[3])

    # 在当前帧上绘制跟踪框
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # 计算FPS
    fps += 1
    if time.time() - curr_time >= 1:
        curr_time = time.time()
        print("FPS: ", fps)

        # 自动更新模板
        update_count += 1
        if update_count == 3:
            update_count = 0
            template = cv2.cvtColor(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])], cv2.COLOR_BGR2GRAY)
            print("Template updated.")

        fps = 0

    # 判断是否丢失目标
    if max_val < 0.3:
        print("Target lost, please select a new one.")
        roi = cv2.selectROI("Frame", frame, False)
        template = cv2.cvtColor(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])], cv2.COLOR_BGR2GRAY)

    # 显示当前帧
    cv2.imshow("Frame", frame)

    # 如果按下ESC键，则退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放资源，关闭窗口
cap.release()
cv2.destroyAllWindows()