#!/usr/bin/env python

#------------------------------------------------------------------------------
#                 PyuEye example - camera modul
#
# Copyright (c) 2017 by IDS Imaging Development Systems GmbH.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#------------------------------------------------------------------------------

import traceback
import numpy as np
from pyueye import ueye
from .pyueye_example_utils import (uEyeException, Rect, get_bits_per_pixel,
                                  ImageBuffer, check)
from .pyueye_example_utils import ImageData

class Camera:
    def __init__(self, device_id=0):
        self.h_cam = ueye.HIDS(device_id)
        self.img_buffers = []
        self.timeout = 1000
        self.color_mode  = None

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, _type, value, traceback):
        self.exit()

    def handle(self):
        return self.h_cam

    def alloc(self, buffer_count=4):
        rect = self.get_aoi()
        bpp = get_bits_per_pixel(self.get_colormode())

        for buff in self.img_buffers:
            check(ueye.is_FreeImageMem(self.h_cam, buff.mem_ptr, buff.mem_id))

        for i in range(buffer_count):
            buff = ImageBuffer()
            ueye.is_AllocImageMem(self.h_cam,
                                  rect.width, rect.height, bpp,
                                  buff.mem_ptr, buff.mem_id)
            
            check(ueye.is_AddToSequence(self.h_cam, buff.mem_ptr, buff.mem_id))

            self.img_buffers.append(buff)

        ueye.is_InitImageQueue(self.h_cam, 0)

    def init(self):
        ret = ueye.is_InitCamera(self.h_cam, None)
        if ret != ueye.IS_SUCCESS:
            self.h_cam = None
            raise uEyeException(ret)
            
        return ret

    def exit(self):
        ret = None
        if self.h_cam is not None:
            ret = ueye.is_ExitCamera(self.h_cam)
        if ret == ueye.IS_SUCCESS:
            self.h_cam = None

    def get_aoi(self):
        rect_aoi = ueye.IS_RECT()
        ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        return Rect(rect_aoi.s32X.value,
                    rect_aoi.s32Y.value,
                    rect_aoi.s32Width.value,
                    rect_aoi.s32Height.value)

    def set_aoi(self, x, y, width, height):
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(x)
        rect_aoi.s32Y = ueye.int(y)
        rect_aoi.s32Width = ueye.int(width)
        rect_aoi.s32Height = ueye.int(height)

        return ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

    def capture_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_CaptureVideo(self.h_cam, wait_param)

    def stop_video(self):
        return ueye.is_StopLiveVideo(self.h_cam, ueye.IS_FORCE_VIDEO_STOP)
    
    def freeze_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_FreezeVideo(self.h_cam, wait_param)

    def set_colormode(self, colormode):
        check(ueye.is_SetColorMode(self.h_cam, colormode))
        self.color_mode = colormode
        
    def get_colormode(self):
        ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_GET_COLOR_MODE)
        return ret

    def get_format_list(self):
        count = ueye.UINT()
        check(ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_GET_NUM_ENTRIES, count, ueye.sizeof(count)))
        format_list = ueye.IMAGE_FORMAT_LIST(ueye.IMAGE_FORMAT_INFO * count.value)
        format_list.nSizeOfListEntry = ueye.sizeof(ueye.IMAGE_FORMAT_INFO)
        format_list.nNumListElements = count.value
        check(ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_GET_LIST,
                                  format_list, ueye.sizeof(format_list)))
        return format_list

    def set_format(self,ID):
        ID = ueye.UINT(ID)
        ueye.is_ImageFormat(self.h_cam, ueye.IMGFRMT_CMD_SET_FORMAT, ID, ueye.sizeof(ID))        
    # ----- EXPOSURE ----- 

    def get_exposure_range(self):
        exposure_min,exposure_max,exposure_inc = ueye.DOUBLE(),ueye.DOUBLE(),ueye.DOUBLE()
        ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, exposure_min, ueye.sizeof(exposure_min))
        ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, exposure_max, ueye.sizeof(exposure_max))
        ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, exposure_inc, ueye.sizeof(exposure_inc))
        return [exposure_min.value,exposure_max.value,exposure_inc.value]
    
    def get_exposure(self):
        exposure = ueye.DOUBLE()
        ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure, ueye.sizeof(exposure))
        return exposure.value
    
    def set_exposure(self,exposure):
        exposure = ueye.double(exposure)
        ueye.is_Exposure(self.h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, ueye.sizeof(exposure))   
    
    def set_gain(self,gain):
        gain = ueye.INT(gain)
        ueye.is_SetHardwareGain(self.h_cam, gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)
        
    def get_gain(self):
        return ueye.is_SetHardwareGain(self.h_cam, ueye.IS_GET_MASTER_GAIN , ueye.IS_IGNORE_PARAMETER,ueye.IS_IGNORE_PARAMETER,ueye.IS_IGNORE_PARAMETER)  
            
    
    # ----- PIXEL CLOCK -----      
    def get_pixel_clock(self):
        pixel_clock = ueye.UINT()
        ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_GET, pixel_clock, ueye.sizeof(pixel_clock))
        return pixel_clock.value
    
    def set_pixel_clock(self,pixel_clock):
        pixel_clock = ueye.UINT(pixel_clock)
        ueye.is_PixelClock(self.h_cam, ueye.IS_PIXELCLOCK_CMD_SET, pixel_clock, ueye.sizeof(pixel_clock))
   
    # ----- FRAME RATE -----
   
    def set_frame_rate(self,frame_rate):
        frame_rate = ueye.double(frame_rate)
        frame_rate_actual = ueye.DOUBLE() 
        ueye.is_SetFrameRate(self.h_cam, frame_rate, frame_rate_actual)
        return frame_rate_actual.value
    
    # ----- EXTERNAL TRIGGER -----
    def enable_external_trigger(self):
        # Hardcoded rising edge
        ueye.is_SetExternalTrigger(self.h_cam, ueye.IS_SET_TRIGGER_LO_HI)
        
    def disable_external_trigger(self):
        # Hardcoded rising edge
        ueye.is_SetExternalTrigger(self.h_cam, ueye.IS_SET_TRIGGER_OFF)    

    def print_formats(self):
        format_list = self.get_format_list()
        for i in range(len(format_list.FormatInfo)):
            print('format ID:{}, {}'.format(format_list.FormatInfo[i].nFormatID,
                                            format_list.FormatInfo[i].strFormatName))
                                            
    def read_frame(self):
        img_buffer = self.img_buffers[0]
        ret = ueye.is_WaitForNextImage(self.h_cam,
                                   self.timeout,
                                   img_buffer.mem_ptr,
                                   img_buffer.mem_id)
        if ret == ueye.IS_SUCCESS:
            image_data = ImageData(self.h_cam, img_buffer)
            frame = image_data.as_1d_image()
            
            # image info
            #try:
            image_info = ueye.UEYEIMAGEINFO()
            ueye.is_GetImageInfo(self.h_cam, img_buffer.mem_id, image_info,ueye.sizeof(image_info))
            timestamp = np.array([image_info.TimestampSystem.wHour.value,
                                  image_info.TimestampSystem.wMinute.value,
                                  image_info.TimestampSystem.wSecond.value,
                                  image_info.TimestampSystem.wMilliseconds.value])
            #except Exception:
            #    traceback.print_exc()

            image_data.unlock()
            return frame,timestamp
        else:
            print('failed')
            
    def set_rolling_shutter(self):
        const = ueye.INT(1)
        ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE,const,ueye.sizeof(const))
        print('Setting to rolling shutter mode')
        
    def set_global_shutter(self):
        const = ueye.INT(2)
        ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE,const,ueye.sizeof(const))
        print('Setting to global shutter mode')

    def set_multi_aoi(self,aoi_list):
        m_nMaxNumberMultiAOIs = ueye.INT()
        ueye.is_AOI(self.h_cam, ueye.IS_AOI_MULTI_GET_AOI | ueye.IS_AOI_MULTI_MODE_GET_MAX_NUMBER, m_nMaxNumberMultiAOIs, ueye.sizeof(m_nMaxNumberMultiAOIs))
        #rMinSizeAOI = ueye.IS_SIZE_2D()
        #ueye.is_AOI(cams[0].h_cam, ueye.IS_AOI_MULTI_GET_AOI | ueye.IS_AOI_MULTI_MODE_GET_MINIMUM_SIZE, rMinSizeAOI, ueye.sizeof(rMinSizeAOI));
        #print(rMinSizeAOI.s32Height)

        pMultiAOIList     = (ueye.IS_MULTI_AOI_DESCRIPTOR * m_nMaxNumberMultiAOIs)
        m_psMultiAOIs =   ueye.IS_MULTI_AOI_CONTAINER(pMultiAOIList)
        m_psMultiAOIs.nNumberOfAOIs = m_nMaxNumberMultiAOIs
        for i in range(len(aoi_list)):
            m_psMultiAOIs.pMultiAOIList[i].nPosX   = aoi_list[i][0]
            m_psMultiAOIs.pMultiAOIList[i].nPosY   = aoi_list[i][1]
            m_psMultiAOIs.pMultiAOIList[i].nWidth  = aoi_list[i][2]
            m_psMultiAOIs.pMultiAOIList[i].nHeight = aoi_list[i][3]
            m_psMultiAOIs.pMultiAOIList[i].nStatus = ueye.IS_AOI_MULTI_STATUS_SETBYUSER

        nRet = ueye.is_AOI(self.h_cam, ueye.IS_AOI_MULTI_SET_AOI,
              m_psMultiAOIs, ueye.sizeof(m_psMultiAOIs) + ueye.sizeof(pMultiAOIList))
        if nRet == ueye.IS_SUCCESS:
        	print('multiple AOIs successfull')
        else:
        	print('error setting multiple AOIs')

    def enable_vertical_aoi_merge_mode(self,line_position,AOI_height,mode=2):
        mode  = ueye.INT(mode)
        ret   = ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_MODE,
            mode, ueye.sizeof(mode))
        if ret == ueye.IS_SUCCESS:
            print('vertical merge ON')

        line_position  = ueye.INT(line_position)
        ret   = ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_POSITION,
                    line_position, ueye.sizeof(line_position))
        if ret == ueye.IS_SUCCESS:
            ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION,
                line_position, ueye.sizeof(line_position))
            print('line position set to {}'.format(line_position))

        height  = ueye.INT(AOI_height)
        ret   = ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_HEIGHT,
                    height, ueye.sizeof(height))
        if ret == ueye.IS_SUCCESS:
            ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT,
                    height, ueye.sizeof(height))
            print('AOI height set to {}'.format(height))   


    def enable_fast_linescan_mode(self,line_number):
        mode  = ueye.INT(ueye.IS_DEVICE_FEATURE_CAP_LINESCAN_MODE_FAST)
        ret   = ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_LINESCAN_MODE,
                    mode, ueye.sizeof(mode))
        if ret == ueye.IS_SUCCESS:
            line_number = ueye.INT(line_number)
            ret = ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_LINESCAN_NUMBER,
                    line_number,ueye.sizeof(line_number))
            ret = ueye.is_DeviceFeature(self.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_LINESCAN_NUMBER,
                    line_number,ueye.sizeof(line_number))
            if ret == ueye.IS_SUCCESS:
                print('line scan number = {}'.format(line_number))
        else:
            print('failed')

'''
nSupportedFeatures = ueye.INT()
ret =  ueye.is_DeviceFeature(cam1.h_cam, ueye.IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_FEATURES,
        nSupportedFeatures,ueye.sizeof(nSupportedFeatures))

mode  = ueye.INT(1)
print(nSupportedFeatures & mode)

line_number = ueye.INT(line_number)
ret = ueye.is_DeviceFeature(cam1.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_LINESCAN_NUMBER,line_number,ueye.sizeof(line_number))
if ret == ueye.IS_SUCCESS:
    print('works')

mode  = ueye.INT(4)
ret   = ueye.is_DeviceFeature(cam1.h_cam, ueye.IS_DEVICE_FEATURE_CMD_SET_LINESCAN_MODE,
            mode, ueye.sizeof(mode))
if ret == ueye.IS_SUCCESS:
    print('works')

'''
   
