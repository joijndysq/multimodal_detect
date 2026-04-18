import cv2

def main():
    # 打开默认摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(2)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头。请检查摄像头连接、权限或是否被其他程序占用。")
        return
    
    print("Camera opened. Press 'q' to quit.")
    
    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        
        # 检查是否成功读取帧
        if not ret:
            print("错误：无法读取视频帧")
            break

        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 放大到 640x640
        gray_resized = cv2.resize(gray, (640, 640), interpolation=cv2.INTER_LINEAR)

        # 显示灰度图
        cv2.imshow('Camera', gray_resized)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("退出程序")
            break
    
    # 释放摄像头资源
    cap.release()
    
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
