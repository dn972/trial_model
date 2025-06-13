import os
import subprocess
import shlex
import argparse
# from Model import *
from utils_VVC import *
import glob, os
import re
from typing import Tuple
from PIL import Image
import torch
from torch import nn
import torchvision.transforms.functional as TF
from torchvision import utils as utils_tv
# from kornia.color import RgbToYuv420, Yuv420ToRgb
import cv2
import numpy as np
import time
from functions.jpeg_torch import *
from compressai.models import *

from compressai.zoo import image_models
from torch.hub import load_state_dict_from_url
from compressai.zoo.pretrained import load_pretrained 
from compressai.models.waseda import Cheng2020Anchor
from compressai.zoo import cheng2020_anchor
# from PyImageProcess.utils_rgb import *
# from PyImageProcess.utils_yuv import *
# import yuvio
def PNG2YUV_sp(PNGFileName:str, SavePath:str):
    # PNG_img = 
    # cmd = strcat("ffmpeg -i ",strcat(inputPath,filename),".jpg -pix_fmt yuv420p ",strcat(outputPath,outputName,".yuv"));
    out_filename = SavePath + "/input_256x256_1.yuv"
    cmd = ['ffmpeg', '-y', '-i', PNGFileName, '-pix_fmt', 'yuv420p', out_filename, "-loglevel", 'quiet']
    # cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_device', '0', '-y', '-i', PNGFileName, '-pix_fmt', 'yuv420p', out_filename]#, "-loglevel", 'quiet']
    # , '-hwaccel_output_format', 'cuda'
    subprocess.call(cmd)
    
def PNG2YUV(PNGFileName:str, SavePath:str):
    # PNG_img = 
    # cmd = strcat("ffmpeg -i ",strcat(inputPath,filename),".jpg -pix_fmt yuv420p ",strcat(outputPath,outputName,".yuv"));
    out_filename = SavePath + "/input_640x426_1.yuv"
    # cmd = ['ffmpeg', '-y', '-i', PNGFileName, '-pix_fmt', 'yuv420p', out_filename, "-loglevel", 'quiet']
    # cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_device', '0', '-y', '-i', PNGFileName, '-pix_fmt', 'yuv420p', out_filename]#, "-loglevel", 'quiet']
    # , '-hwaccel_output_format', 'cuda'
    # subprocess.call(cmd)
    cmd = 'ffmpeg -y -i ' + PNGFileName + ' -pix_fmt yuv420p ' + out_filename + ' -loglevel quiet'
    # print(cmd)
    os.system(cmd)

def PNG2YUV_h(PNGFileName:str, SavePath:str):
    # PNG_img = 
    # cmd = strcat("ffmpeg -i ",strcat(inputPath,filename),".jpg -pix_fmt yuv420p ",strcat(outputPath,outputName,".yuv"));
    out_filename = SavePath + "/inputh_256x256_1.yuv"
    # cmd = ['ffmpeg', '-y', '-i', PNGFileName, '-pix_fmt', 'yuv420p', out_filename, "-loglevel", 'quiet']
    # cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_device', '0', '-y', '-i', PNGFileName, '-pix_fmt', 'yuv420p', out_filename]#, "-loglevel", 'quiet']
    # , '-hwaccel_output_format', 'cuda'
    # subprocess.call(cmd)
    cmd = 'ffmpeg -y -i ' + PNGFileName + ' -pix_fmt yuv420p ' + out_filename + ' -loglevel quiet'
    # print(cmd)
    os.system(cmd)
def YUV2PNG_sp(YUVFileName:str, out_filename:str):
    # BQSquare_416x240_60
    # out_filename = SavePath + "/output.png"
    # cmd = ['ffmpeg', '-f', 'rawvideo', '-framerate', '1', '-video_size', '640x510', '-pix_fmt', 'yuv420p', 'dst_range', '1', '-i', YUVFileName, '-pix_fmt', 'rgb24', "output.png"]
    cmd = ['ffmpeg', '-y','-f', 'image2', '-c:v', 'rawvideo',  '-framerate', '1', '-video_size', f'256x256', '-pix_fmt', 'yuv420p', '-i', YUVFileName, '-pix_fmt', 'rgb24', out_filename, "-loglevel", 'quiet']
    # cmd = ['ffmpeg', '-hwaccel', 'cuvid', '-hwaccel_output_format', 'cuda', '-y','-f', 'image2', '-c:v', 'rawvideo',  '-framerate', '1', '-video_size', f'256x256', '-pix_fmt', 'yuv420p', '-i', YUVFileName, '-pix_fmt', 'rgb24', out_filename, "-loglevel", 'quiet']
    # '-dst_range', '1',
    # print(YUVFileName)
    subprocess.call(cmd)  

def YUV2PNG(YUVFileName:str, out_filename:str):
    
    cmd = 'ffmpeg -y -f image2 -c:v rawvideo -framerate 1 -video_size 640x426 -pix_fmt yuv420p -i ' + YUVFileName + ' -pix_fmt rgb24 ' + out_filename + ' -loglevel quiet'
    # print(cmd)
    os.system(cmd)


#Define
# CODING_SCHEME_CFG_NAME = "randomaccess_fast"
# CODING_SCHEME_CFG_NAME = "vvenc_all_intra"
CODING_SCHEME_CFG_NAME = "encoder_intra_vtm"
INPUT_BIT_DEPTH = 8
INPUT_CHROMA_FORMAT = 420
NUM_FRAMES = 1
METHOD = "OrgUbuntu"
BITSTREAM_FOLDER = "/media/EXT0/daole/sd_scripts/VVC_Python/"
EXECUTABLE_FILE_NAME = BITSTREAM_FOLDER + "EncoderApp" 
width = 256
height = 256
frameRate = 1


CMD_LIST = []
def Encoder_VVC(YUVFileName:str, SavePath:str):
    SavePath = SavePath + "/"
    for qpIndex, qp in enumerate(QP_BASE_LIST):
        # outFileName = METHOD + "_" + CODING_SCHEME_CFG_NAME + "_" + "x0_yuv" + "_" + "QP" + str(qp)
        outFileName = "rec_QP" + str(qp) + "_256x256_1" 

        cmd = EXECUTABLE_FILE_NAME
        cmd = Ultility.addParams(cmd, "-c", BITSTREAM_FOLDER +"cfg/" + CODING_SCHEME_CFG_NAME + ".cfg")
        #cmd = Ultility.addParams(cmd, "--threads="+str(THREADS),"")


        cmd = Ultility.addParams(cmd, "--InputFile="+ "\""+YUVFileName+ "\"", "")
        # cmd = Ultility.addParams(cmd, "-wdt", str(width))
        # cmd = Ultility.addParams(cmd, "-hgt", str(height))
        cmd = Ultility.addParams(cmd, "--size="+"256x256", "")
        cmd = Ultility.addParams(cmd, "--InputBitDepth="+str(INPUT_BIT_DEPTH), "")
        cmd = Ultility.addParams(cmd, "--InputChromaFormat="+str(INPUT_CHROMA_FORMAT), "")
        cmd = Ultility.addParams(cmd, "-fr", str(frameRate))
        # cmd = Ultility.addParams(cmd, "--FramesToBeEncoded="+str(seqInfo.numFrames), "")
        cmd = Ultility.addParams(cmd, "--FramesToBeEncoded="+str(1), "")
        cmd = Ultility.addParams(cmd, "--QP="+str(qp), "")
        cmd = Ultility.addParams(cmd, "-b", SavePath + outFileName + ".h266")
        cmd = Ultility.addParams(cmd, "--ReconFile="+str(SavePath)  + outFileName + ".yuv", "")
        cmd = Ultility.addParams(cmd, "--OutputBitDepth="+str(INPUT_BIT_DEPTH), "")
        cmd = Ultility.addParams(cmd, "--OutputBitDepth="+str(INPUT_BIT_DEPTH), "")
        pathLogFile = SavePath + outFileName + ".txt"

        # CMD_LIST.append((cmd))
        CMD_LIST.append((cmd, pathLogFile))
        sub_process.call
        # print(cmd)
    # end loop

    Ultility.runIntensiveTask(CMD_LIST, 1)

def Encoder_VVC_Single_sp(YUVFileName:str, SavePath:str):
    SavePath = SavePath + "/"
    outFileName = "rec_QP52_256x256_1" 
    cfg_filename = BITSTREAM_FOLDER + "cfg/"+ CODING_SCHEME_CFG_NAME+".cfg"
    bin_filename = SavePath + outFileName + ".h266"
    rec_filename = SavePath + outFileName + ".yuv"

    cmd = [EXECUTABLE_FILE_NAME, '-c', cfg_filename, '-i', YUVFileName, '-s', '256x256', '--InputBitDepth=8', '--InputChromaFormat=420', '-fr', str(frameRate), '--FramesToBeEncoded=1', '--QP=52', '-b', bin_filename, f'--ReconFile='+rec_filename, '--OutputBitDepth=8']
    # print(cmd)
    subprocess.call(cmd, stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT)

def Encoder_VVC_Single(YUVFileName:str, SavePath:str):
    SavePath = SavePath + "/"
    outFileName = "rec_QP52_640x426_1" 
    cfg_filename = BITSTREAM_FOLDER + "cfg/"+ CODING_SCHEME_CFG_NAME+".cfg"
    bin_filename = SavePath + outFileName + ".h266"
    rec_filename = SavePath + outFileName + ".yuv"

    # cmd = EXECUTABLE_FILE_NAME + ' -c ' + cfg_filename + ' --InputFile=' + YUVFileName + ' -s 640x426 --InputBitDepth=8 --InputChromaFormat=420 -fr ' + str(frameRate) + ' --FramesToBeEncoded=1 --QP=52 --ReconFile=' +rec_filename+ ' --OutputBitDepth=8' #' > /dev/null 2>&1' ##vvenc
    cmd = EXECUTABLE_FILE_NAME + ' -c ' + cfg_filename + ' --InputFile=' + YUVFileName + ' --SourceWidth=640 --SourceHeight=426 --InputBitDepth=8 --InputChromaFormat=420 --FrameRate=' + str(frameRate) + ' --FramesToBeEncoded=1 --QP=52 --ReconFile=' +rec_filename+ ' --OutputBitDepth=8' #' > /dev/null 2>&1'
    print(cmd)
    os.system(cmd)

def Encoder_VVC_Single_h(YUVFileName:str, SavePath:str):
    SavePath = SavePath + "/"
    outFileName = "rech_QP52_256x256_1" 
    cfg_filename = BITSTREAM_FOLDER + "cfg/"+ CODING_SCHEME_CFG_NAME+".cfg"
    bin_filename = SavePath + outFileName + ".h266"
    rec_filename = SavePath + outFileName + ".yuv"

    cmd = EXECUTABLE_FILE_NAME + ' -c ' + cfg_filename + ' -i ' + YUVFileName + ' -s 256x256 --InputBitDepth=8 --InputChromaFormat=420 -fr ' + str(frameRate) + ' --FramesToBeEncoded=1 --QP=52 --ReconFile=' +rec_filename+ ' --OutputBitDepth=8' ' > /dev/null 2>&1'
    # print(cmd)
    os.system(cmd)

def Encoder_VVC_woSaving(Y, UV, SavePath:str):
    SavePath = SavePath + "/"
    outFileName = "rec_QP52_256x256_1" 
    cfg_filename = BITSTREAM_FOLDER + "cfg/"+ CODING_SCHEME_CFG_NAME+".cfg"
    bin_filename = SavePath + outFileName + ".h266"
    rec_filename = SavePath + outFileName + ".yuv"

    cmd = EXECUTABLE_FILE_NAME + ' -c ' + cfg_filename + ' -i ' + YUVFileName + ' -s 256x256 --InputBitDepth=8 --InputChromaFormat=420 -fr ' + str(frameRate) + ' --FramesToBeEncoded=1 --QP=52 --ReconFile=' +rec_filename+ ' --OutputBitDepth=8' 
    # print(cmd)
    os.system(cmd)

def Encoder_VVC_Single_QP49(YUVFileName:str, SavePath:str):
    SavePath = SavePath + "/"
    outFileName = "rec_QP52_256x256_1" 
    cfg_filename = BITSTREAM_FOLDER + "cfg/"+ CODING_SCHEME_CFG_NAME+"_QP49.cfg"
    bin_filename = SavePath + outFileName + ".h266"
    rec_filename = SavePath + outFileName + ".yuv"

    cmd = EXECUTABLE_FILE_NAME + ' -c ' + cfg_filename + ' -i ' + YUVFileName + ' -s 256x256 --InputBitDepth=8 --InputChromaFormat=420 -fr ' + str(frameRate) + ' --FramesToBeEncoded=1 --QP=52 --ReconFile=' +rec_filename+ ' --OutputBitDepth=8 > /dev/null 2>&1'
    # print(cmd)
    os.system(cmd)
def vvc_func(x_in, t, SavePath:str):
    # filename = os.path.join(SavePath, f"{str(i).zfill(6)}.png")
    utils_tv.save_image(x_in, os.path.join(SavePath, f"input_{str(t)}.png"), nrow=1, normalize=True)
    # start_time = time.time()
    PNG2YUV(os.path.join(SavePath, f"input_{str(t)}.png"), SavePath)
    # print("--- %s seconds for png2yuv ---" % (time.time() - start_time))
    Encoder_VVC_Single(os.path.join(SavePath, f"input_256x256_1.yuv"), SavePath)
    # print("--- %s seconds for Encoder_VVC ---" % (time.time() - start_time))
    output_filename = os.path.join(SavePath, f"output{str(t)}.png")
    YUV2PNG(os.path.join(SavePath, f"rec_QP52_256x256_1.yuv"), output_filename)
    # print("--- %s seconds for YUV2PNG ---" % (time.time() - start_time))
    x_in_lr = Image.open(os.path.join(SavePath, f"output{str(t)}.png"))               
    # To TENSOR
    x_in_lr = TF.to_tensor(x_in_lr).to(x_in.device)

    return x_in_lr

def vvc_func_test(x_in, t, SavePath:str):
    # filename = os.path.join(SavePath, f"{str(i).zfill(6)}.png")
    utils_tv.save_image(x_in, os.path.join(SavePath, f"input_{str(t)}.png"), nrow=1, normalize=True)
    # start_time = time.time()
    PNG2YUV(os.path.join(SavePath, f"input_{str(t)}.png"), SavePath)
    # print("--- %s seconds for png2yuv ---" % (time.time() - start_time))
    Encoder_VVC_Single(os.path.join(SavePath, f"input_640x426_1.yuv"), SavePath)
    # print("--- %s seconds for Encoder_VVC ---" % (time.time() - start_time))
    output_filename = os.path.join(SavePath, f"output{str(t)}.png")
    YUV2PNG(os.path.join(SavePath, f"rec_QP52_640x426_1.yuv"), output_filename)
    # print("--- %s seconds for YUV2PNG ---" % (time.time() - start_time))
    x_in_lr = Image.open(os.path.join(SavePath, f"output{str(t)}.png"))               
    # To TENSOR
    x_in_lr = TF.to_tensor(x_in_lr).to(x_in.device)

    return x_in_lr

def vvc_func_h(x_in, t, SavePath:str):
    # filename = os.path.join(SavePath, f"{str(i).zfill(6)}.png")
    utils_tv.save_image(x_in, os.path.join(SavePath, f"inputh_{str(t)}.png"), nrow=1, normalize=False)
    # start_time = time.time()
    PNG2YUV_h(os.path.join(SavePath, f"inputh_{str(t)}.png"), SavePath)
    # print("--- %s seconds for png2yuv ---" % (time.time() - start_time))
    Encoder_VVC_Single_h(os.path.join(SavePath, f"inputh_256x256_1.yuv"), SavePath)
    # print("--- %s seconds for Encoder_VVC ---" % (time.time() - start_time))
    output_filename = os.path.join(SavePath, f"outputh{str(t)}.png")
    YUV2PNG(os.path.join(SavePath, f"rech_QP52_256x256_1.yuv"), output_filename)
    # print("--- %s seconds for YUV2PNG ---" % (time.time() - start_time))
    x_in_lr = Image.open(os.path.join(SavePath, f"outputh{str(t)}.png"))      
    # print(x_in_lr.min())
    # print(x_in_lr.max())         
    # To TENSOR
    x_in_lr = TF.to_tensor(x_in_lr).to(x_in.device)

    return x_in_lr

def compressai_func(x_in):
    IFrameCompressor = JointAutoregressiveHierarchicalPriors(N=192, M=192)
    # IFrameCompressor = Cheng2020Anchor(N = 128)
    IFrameCompressor = IFrameCompressor.to(x_in.device)
    # for p in IFrameCompressor.parameters():
    #     p.requires_grad = False
    # url = "https://compressai.s3.amazonaws.com/models/v1/cheng2020-anchor-1-dad2ebff.pth.tar"
    url = "https://compressai.s3.amazonaws.com/models/v1/mbt2018-1-3f36cd77.pth.tar"
    checkpoint = load_state_dict_from_url(url, progress=True, map_location=x_in.device)
    checkpoint = load_pretrained(checkpoint)
    IFrameCompressor.load_state_dict(checkpoint)
    x_hat = IFrameCompressor(x_in)
    return x_hat["x_hat"]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
    def forward(self, x):
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # transform the input
        # print('x', x.type())
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

def main():
    # x = torch.randn(1, 1, 28, 28).float()
    # # x = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    # print(x)
    # net = Net()
    # y = net(x)
    # print(y)
    # print('L', y - x)
    # CNet = ssf2020('1', metric='mse', pretrained=True, progress=True)
    path = '000000_org.png'
    x_in = Image.open(path)
    
    # # To TENSOR
    x_in = TF.to_tensor(x_in)[None, :]
    vvc_func_test(x_in, 0, '/media/EXT0/daole/sd_scripts/VVC_Python/')
   
if __name__ == "__main__":
    main()
