import cv2
import numpy as np

# 在第一帧手动选取跟踪目标的位置
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('frame', frame)
x, y, w, h = cv2.selectROI(frame)
roi = frame[y:y+h, x:x+w]

# 用选取的目标模板初始化卡尔曼滤波器
kalman = cv2.KalmanFilter(4, 2, 0)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
kalman.errorCovPost = np.eye(4, dtype=np.float32)

while True:
    # 读取下一帧图像
    ret, frame = cap.read()

    # 使用模板匹配确定目标位置
    res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    print(x,y)
    print("--------------------")
    # 使用卡尔曼滤波器对观测结果进行修正
    measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.predict()
    kalman.correct(measurement)

    # 获取修正后的跟踪目标位置
    predicted_state = kalman.predict()

    # 在图像中绘制当前跟踪结果
    center = (int(predicted_state[0]), int(predicted_state[1]))
    cv2.circle(frame, center, 20, (0, 0, 255), 2)
    cv2.rectangle(frame, center, (center[0] + w, center[1] + h), (0, 0, 255), 2)

    # 显示图像并等待按键
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # 更新下一帧的跟踪目标模板
    roi = frame[y:y+h, x:x+w]

cap.release()
cv2.destroyAllWindows()