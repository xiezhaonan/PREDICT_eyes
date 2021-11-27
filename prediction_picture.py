import torch
import cv2
from option import BasciOption
import dlib
import numpy as np
import time
from models import GoogleNet3
from models import create_transform

if __name__ == '__main__':
    opt = BasciOption(train_flag=False).initialize()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogleNet3()
    offset_pixelY = -40
    offset_pixe2Y = 0
    offset_pixelX = -40
    offset_pixe2X = 40
    model.load_state_dict(torch.load("./Basic_Epoch_3_Accuracy_0.93.pth"))
    model = model.to(device)
    transformer = create_transform(opt)
    predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
    video_size = (200, 50)  # w h
    offset_pixelY = -14
    offset_pixelX = 0
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    fps = 0.0
    face_offset = 100
    t1 = time.time()
    frame = cv2.imread("./a0eef8c53337cf3ac3ca50ebe007d4a9.jpg")
    dets = detector(frame, 0)
    pt_pos = []
    eye_w, eye_h = 40, 30
    if dets:
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            for index, pt in enumerate(shape.parts()):
                pt_pos.append((pt.x, pt.y))
            w = abs(pt_pos[27][1] - face_offset - pt_pos[8][1])
            h = abs(pt_pos[0][0] - pt_pos[14][0])
            video_size = (w, h)
            face = frame[pt_pos[27][1] - face_offset:pt_pos[8][1], pt_pos[0][0]:pt_pos[14][0]]
            try:
                face = cv2.resize(face, video_size)
            except cv2.error:
                continue
            left_eye = frame[
                       pt_pos[37][1] + offset_pixelY:pt_pos[37][1]
                                                          + eye_h + offset_pixelY,
                       pt_pos[36][0] + offset_pixelX:pt_pos[36][
                                                              0] + eye_w + offset_pixelX]

            right_eye = frame[
                        pt_pos[44][1] + offset_pixelY:pt_pos[44][1]
                                                           + eye_h + offset_pixelY,
                        pt_pos[42][0] + offset_pixelX:pt_pos[42][
                                                               0] + eye_w + offset_pixelX]

            crop_eye = np.concatenate((left_eye, right_eye),
                                      axis=1)
            cv2.imshow("eyes",crop_eye)
            inputs = transformer(crop_eye).to(device)
            outputs = model(inputs.unsqueeze(0))
            _,y_pred = torch.max(outputs,dim = 1)
            frame = cv2.putText(frame, "label= %.2f" % (y_pred), (0, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    fps = (fps + (1. / (time.time() - t1))) / 2
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('origin_pic', frame)
    cv2.waitKey(0)