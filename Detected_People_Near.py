#############################################################
## 인공지능과 기계학습
##  조이름  : 소심조 (1조)
##  작품명  : 초록안전 시스템
## 작품설명 : 보행자를 인식하여 가까이 오면 경고음을 발생
## 실행환경 : Python 3.7.7
##            CUDA 10.1
##            cuDNN v7.6.5
##            OpenCV 4.3.0
##            Yolo v4
##            Yolo>darknet>build>darknet>x64에서 실행
##            Input과 Output 폴더를 생성해서 데이터입력과 저장
##############################################################

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations
import winsound as ws

## 호출시 비프음을 발생시킨다.
def beepsound():
    freq = 2000        ## range : 37 ~ 32767
    dur = 100          ## ms
    ws.Beep(freq, dur) ## winsound.Beep(frequency, duration)
    ## print(beepsound())

## y축의 max와 min을 입력받아 크기를 계산
def is_close(ymin, ymax):
    dst = round(abs(ymax-ymin), 1) # 소수점 1자리까지 계산
    return dst # 

## 중심 좌표를 직사각형 좌표로 변환한다.
def convertBack(x, y, w, h): 
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    :param:
    detections = 한 프레임에서 검출된 모든 객체
    img = darknet(YOLO)을 사용한 이미지파일

    :return:
    객체를 박스처리 한 이미지파일
    """
    #================================================================
    # 3.1 Purpose : 탐지에서 Persons 클래스를 필터링하고 각 사람 감지에 대한 경계 상자 중심을 가져온다.
    #================================================================
    if len(detections) > 0:  						# 이미지에서 최소 1 회 감지 및 프레임에서 감지 유무 확인
        centroid_dict = dict() 						# 함수는 사전을 만들고이를 centroid_dict라고 부릅니다.
        objectId = 0							# ObjectId라는 변수를 초기화하고 0으로 설정합니다
        for detection in detections:				        # 이 if 문에서 사람에 대한 모든 탐지를 필터링합니다.
            # 태그가 사람일 경우만 확인 
            name_tag = str(detection[0].decode())
            if name_tag == 'person':
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	# 탐지된 객체의 중심점 저장
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # 중심 좌표에서 직사각형 좌표로 변환, 부동 소수점을 사용하여 BBox의 정밀도를 보장한다.            
                # 탐지 된 사람의 bbox 중심점을 추가
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # bbox의 중심점과 'objectId'를 사용하여 튜플 사전 만들기
                objectId += 1 #각 탐지에 대한 갯수 증가      
    #=================================================================#
    
    #=================================================================
    # 3.2 Purpose : 크기 계산
    #=================================================================            	
        red_zone_list = [] # 임계 값 조건에있는 오브젝트 ID를 포함하는 목록
        for idx, box in centroid_dict.items(): # 감지된 모든 보행자의 정보
            distance = is_close(box[5], box[3]) 			# 크기 계산
            if distance > 40.0:						# 일정 크기 이상이라면
                if idx not in red_zone_list:
                    red_zone_list.append(idx)       #  리스트에 id를 추가한다.
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in red_zone_list:   # red zone list에 id가 있다면
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # 빨간색 경계 상자 만들기
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # 초록색 경계 상자 만들기
	#=================================================================#

	#=================================================================
    	# 3.3 Purpose : 정보 표시
    	#=================================================================        
        text_detect = "Detect object : %s" % str(len(detections))        # 전체 감지된 숫자 
        text_people = "Detect People : %s" % str(len(centroid_dict))    # 감지된 보행자 숫자
        text_near = "Detect Near People : %s" % str(len(red_zone_list)) # 가까운 곳에 있는 보행자 숫자
        # 문자열 생성 위치
        location_detect = (10,25)
        location_people = (10,50)
        location_near = (10,75)						    					
                # 이미지   문자열           좌표              폰트형태          크기     색상     두께  선 종류
        cv2.putText(img, text_detect, location_detect, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246,86,86), 2, cv2.LINE_AA)  # Display Text
        cv2.putText(img, text_people, location_people, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246,86,86), 2, cv2.LINE_AA)  # Display Text
        cv2.putText(img, text_near, location_near, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246,86,86), 2, cv2.LINE_AA)      # Display Text
        #=================================================================#
        # 3.4 Purpose : 가까운 곳에 보행자가 있다면 비프음 발생
        #=================================================================
        if red_zone_list: # 리스트가 비어있지 않다면
            print(beepsound())
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg" # 환경설정파일 경로
    weightPath = "./yolov4.weights" # 학습된 모델 경로
    metaPath = "./cfg/coco.data"    # 코코 데이터 경로
    # 각 경로에 파일이 존재하지 않으면
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    # 파일 열기
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0) 웹캠 사용시 주석해제
    for i in range(9):
        cap = cv2.VideoCapture("./Input/"+str(i+1)+".mp4") # 1~9.mp4를 입력받는다
        frame_width = int(cap.get(3))  # 영상의 너비
        frame_height = int(cap.get(4)) # 영상의 높이
        new_height, new_width = frame_height // 2, frame_width // 2 
        # print("Video Reolution: ",(width, height)) #해상도 출력

        out = cv2.VideoWriter( # 결과물 저장
                "./Output/"+str(i+1)+".avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                (new_width, new_height))
        
        # print("Starting the YOLO loop...")

        # 탐지 할 때마다 재사용하는 이미지를 만든다.
        darknet_image = darknet.make_image(new_width, new_height, 3)
        
        while True:
            prev_time = time.time()
            ret, frame_read = cap.read()
            # 프레임이 존재하면 true가 반환, 없으면 루프가 중단된다.
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB) # BGR->GrayScale 변환
            frame_resized = cv2.resize(frame_rgb, #프레임 크기 재조절
                                       (new_width, new_height),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes()) # 이미지를 바이트형식으로 가져옴

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.75) # darknet을 이용한 객체탐지, 정확도 75% 이하는 버림
            image = cvDrawBoxes(detections, frame_resized) # 네모박스 그리기
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 색상 전환
            print(1/(time.time()-prev_time)) # 진행시간 확인
            cv2.imshow('Detection near people', image) # 이미지 출력
            cv2.waitKey(3) # 키보드 입력 대기
            out.write(image) # 이미지파일 저장

        cap.release() # cap객체 할당해제
        out.release() # out객체 할당해제
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
