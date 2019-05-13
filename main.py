import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Value, Manager
from ctypes import c_bool
from moviepy.editor import *
#import config.py
import config

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
capture_images = []


#비디오 재생 함수
def show_video(video_is_finished, capture_is_finished, video_is_ready, capture_is_ready):
    #video_to_show = cv2.VideoCapture(config.info_video_title)
    #fps = video_to_show.get(cv2.CAP_PROP_FPS)
    clip = VideoFileClip(config.info_video_title)
    #원하는 크기로 resize
    myclip = clip.resize(0.5)
    #프로세스 준비 완료를 알림
    video_is_ready.value = True
    print('video_is_ready')
    #웹캠 준비까지 대기
    while not (video_is_ready.value and capture_is_ready.value):
         cv2.waitKey(1)
    myclip.preview()
    #풀스크린으로 보고싶다면
    #myclip.preview(fullscreen=True)

    #비디오 종료를 알림
    print('video is finished')
    video_is_finished.value = True

    #previous version(opencv)
    '''
    while True:
        now = time.time()
        ret, frame = video_to_show.read()
        if ret:
            cv2.imshow("VideoFrame", cv2.resize(frame, (960, 457)))
        else:
            print('video is finished')
            video_is_finished.value = True
            print(video_is_finished.value)
            video_to_show.release()
            cv2.destroyAllWindows()
            return 0

        #if cv2.waitKey(1) & 0xFF == ord('q'): break
        #if cv2.waitKey(32) > 0: break
        cv2.waitKey(int(1000/fps))
        print(time.time() - now)

    video_to_show.release()
    cv2.destroyAllWindows()
    '''


#웹캠 캡쳐 함수
def capture_webcam(video_is_finished, capture_is_finished, video_is_ready, capture_is_ready, capture_images):
    #웹캠 준비
    video_webcam = cv2.VideoCapture(0)
    count = 0
    #웹캠 준비 완료를 알림
    capture_is_ready.value = True
    print('capture_is_ready')
    #비디오 준비가 끝날때까지 대기
    while not (video_is_ready.value and capture_is_ready.value):
        cv2.waitKey(1)

    #비디오 재생이 끝날때까지 웹캠 캡쳐
    while not video_is_finished.value:
        now = time.time()
        for frame_count in range(config.capture_frame_per_half_sec):
            ret, frame = video_webcam.read()
            '''
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)
            
            for face_i in faces:
                x, y, w, h = face_i
                original_path = config.info_name + '/' + config.path_original + str(count) + '/' + str(frame_count) + '.jpg'
                cropped_path = config.info_name + '/' + config.path_original + str(count) + '/' + str(frame_count) + '.jpg'
                cropped_gray_path = config.info_name + '/' + config.path_original\
                                + str(count) + '/' + str(frame_count) + '.jpg'
                capture_images.append([original_path, frame])
                #cv2.imwrite(config.info_name + '/' + config.path_original + 'aaa.jpg', frame)
                face_crop = frame[y:y + h, x:x + w]
                capture_images.append([cropped_path, face_crop])
                #cv2.imwrite(config.info_name + '/' + config.path_cropped_rgb + 'bbb.jpg', face_crop)
                gray_face_crop = cv2.resize(face_crop, (48, 48))
                gray_face_crop = cv2.cvtColor(gray_face_crop, cv2.COLOR_BGR2GRAY)
                gray_face_crop = gray_face_crop.astype('float32') / 255
                gray_face_crop = np.asarray(gray_face_crop)
                capture_images.append([cropped_gray_path, gray_face_crop])
                #cv2.imwrite(config.info_name + '/' + config.path_cropped_rgb + 'bbb.jpg', gray_face_crop)
            '''
            #이미지를 리스트에 저장
            path = str(count) + '/' + str(frame_count) + '.jpg'
            capture_images.append([path, frame])
            #캡쳐 프레임만큼 대기
            cv2.waitKey(500 // config.capture_frame_per_half_sec)

        print(time.time() - now)
        count += 1

    #캡쳐 종료를 알림
    capture_is_finished.value = True
    print('capture is finished', count)
    #cv 릴리즈
    video_webcam.release()
    cv2.destroyAllWindows()


#이미지 저장 함수
def save_images(img_list):
    print('save start!')
    for path, img in img_list:
        #저장 패스를 지정하고 폴더가 없으면 생성
        original_path = config.info_name + '/' + config.path_original + path
        if not os.path.isdir(os.path.dirname(original_path)):
            os.makedirs(os.path.dirname(original_path))
        cropped_path = config.info_name + '/' + config.path_cropped_rgb + path
        if not os.path.isdir(os.path.dirname(cropped_path)):
            os.makedirs(os.path.dirname(cropped_path))
        cropped_gray_path = config.info_name + '/' + config.path_cropped_grayscale_resized_48 + path
        if not os.path.isdir(os.path.dirname(cropped_gray_path)):
            os.makedirs(os.path.dirname(cropped_gray_path))

        #패스 출력
        print(original_path)
        print(cropped_path)
        print(cropped_gray_path)
        #이미지 그레이 스케일 및 페이스 디텍션
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)
        for face_i in faces:
            x, y, w, h = face_i
            #원본
            cv2.imwrite(original_path, img)
            # 원본 + 얼굴 디텍션, 크롭
            face_crop = img[y:y + h, x:x + w]
            cv2.imwrite(cropped_path, face_crop)
            #흑백 + 얼굴 디텍션, 크롭 + 리사이즈 + 학습을 위한 노말라이즈
            gray_face_crop = cv2.resize(face_crop, (48, 48))
            gray_face_crop = cv2.cvtColor(gray_face_crop, cv2.COLOR_BGR2GRAY)
            gray_face_crop = gray_face_crop.astype('float32') / 255
            gray_face_crop = np.asarray(gray_face_crop)
            cv2.imwrite(cropped_gray_path, gray_face_crop)
    print('save is done')


if __name__ == "__main__":
    #멀티 프로세스 공용 리스트를 위한 매니저
    with Manager() as manager:
        result = Queue()
        #이미지 저장 리스트
        capture_images = manager.list([])
        #프로세스간 상태 공유를 위한 bool변수
        video_is_finished = Value(c_bool, False)
        capture_is_finished = Value(c_bool, False)
        video_is_ready = Value(c_bool, False)
        capture_is_ready = Value(c_bool, False)
        #캡쳐와 재생 싱크를 맞추기 위한 멀티 프로세스 동작
        pc1 = Process(target=show_video, args=(video_is_finished, capture_is_finished, video_is_ready, capture_is_ready))
        pc2 = Process(target=capture_webcam, args=(video_is_finished, capture_is_finished,
                                                   video_is_ready, capture_is_ready, capture_images))
        #프로세스 시작 및 종료시까지 대기
        pc1.start()
        pc2.start()
        pc1.join()
        pc2.join()
        #비디오 재생, 캡쳐 종료 시 이미지 저장
        save_images(capture_images)
        #끝
        print('All process is done!')