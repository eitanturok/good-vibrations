import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

def adjustGamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def convert_frame_to_uint8(frame):
    # frame is debayered
    if frame.dtype=='uint8':
        return frame
    else:
        return np.floor_divide(frame,256).astype('uint8')

def debayer_sequence(seq,mode='BGR'):
    if mode =='BGR':
        mode = cv2.COLOR_BAYER_BG2BGR
    else:
        mode = cv2.COLOR_BAYER_BG2RGB
        
    seq_color = np.zeros((seq.shape[0],seq.shape[1],seq.shape[2],3),dtype=seq.dtype)
    for i in range(seq.shape[0]):
        seq_color[i] = cv2.cvtColor(seq[i],mode)
    return seq_color

def display_16bit_BG(frame,is_out_RGB=1,gamma=1.4):
    frame = convert_frame_to_uint8(frame)
    if len(frame.shape)==2:
        frame = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR)
    if is_out_RGB:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = adjustGamma(frame,gamma)
    return frame

def bgr2rgb(img):
    return img[...,::-1].copy()