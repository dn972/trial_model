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
from kornia.color import RgbToYuv420, Yuv420ToRgb
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
    out_filename = SavePath + "/input_256x256_1.yuv"
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
    
    cmd = 'ffmpeg -y -f image2 -c:v rawvideo -framerate 1 -video_size 256x256 -pix_fmt yuv420p -i ' + YUVFileName + ' -pix_fmt rgb24 ' + out_filename + ' -loglevel quiet'
    # print(cmd)
    os.system(cmd)


#Define
CODING_SCHEME_CFG_NAME = "randomaccess_fast"
INPUT_BIT_DEPTH = 8
INPUT_CHROMA_FORMAT = 420
NUM_FRAMES = 1
METHOD = "OrgUbuntu"
BITSTREAM_FOLDER = "/media/EXT0/daole/sd_scripts/VVC_Python/"
EXECUTABLE_FILE_NAME = BITSTREAM_FOLDER + "vvencFFapp" 
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
        pathLogFile = SavePath + outFileName + ".txt"

        # CMD_LIST.append((cmd))
        # CMD_LIST.append((cmd, pathLogFile))
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
    outFileName = "rec_QP52_256x256_1" 
    cfg_filename = BITSTREAM_FOLDER + "cfg/"+ CODING_SCHEME_CFG_NAME+".cfg"
    bin_filename = SavePath + outFileName + ".h266"
    rec_filename = SavePath + outFileName + ".yuv"

    cmd = EXECUTABLE_FILE_NAME + ' -c ' + cfg_filename + ' -i ' + YUVFileName + ' -s 256x256 --InputBitDepth=8 --InputChromaFormat=420 -fr ' + str(frameRate) + ' --FramesToBeEncoded=1 --QP=52 --ReconFile=' +rec_filename+ ' --OutputBitDepth=8' ' > /dev/null 2>&1'
    # print(cmd)
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
def main():
    # CNet = ssf2020('1', metric='mse', pretrained=True, progress=True)
    path = '000000.png'
    x_in = Image.open(path)
    # To TENSOR
    x_in = TF.to_tensor(x_in)[None, :]
    # x = torch_rgb2ycbcr(x_in)
    # x_luma, x_chroma = chroma_subsample(x)
    # image_models.(1, metric='mse', pretrained=True, progress=True)
    # "bmshj2018-factorized": FactorizedPrior,
    # "bmshj2018_factorized_relu": FactorizedPriorReLU,
    # "bmshj2018-hyperprior": ScaleHyperprior,
    # "mbt2018-mean": MeanScaleHyperprior,

    # IFrameCompressor = FactorizedPrior(N=128, M=192)#bmshj2018_factorized
    # # url = "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-1-446d5c7f.pth.tar"#mse
    # url = "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-ms-ssim-1-9781d705.pth.tar"#ssim
    
    IFrameCompressor = ScaleHyperprior(N=128, M=192)#bmshj2018-hyperprior
    url = "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-1-7eb97409.pth.tar"#mse
    # url = "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-ms-ssim-1-5cf249be.pth.tar"#ssim
    

    # IFrameCompressor = MeanScaleHyperprior(N=128, M=192)#mbt2018-mean
    # # url = "https://compressai.s3.amazonaws.com/models/v1/mbt2018-mean-1-e522738d.pth.tar"#mse
    # url = "https://compressai.s3.amazonaws.com/models/v1/mbt2018-mean-ms-ssim-1-5bf9c0b6.pth.tar"#ssim
    
    # IFrameCompressor = JointAutoregressiveHierarchicalPriors(N=192, M=192)#mbt2018
    # url = "https://compressai.s3.amazonaws.com/models/v1/mbt2018-1-3f36cd77.pth.tar"
    
    # IFrameCompressor = Cheng2020Anchor(N = 128)#cheng2020-anchor
    # # url = "https://compressai.s3.amazonaws.com/models/v1/cheng2020-anchor-1-dad2ebff.pth.tar"  #mse
    # url = "https://compressai.s3.amazonaws.com/models/v1/cheng2020_anchor-ms-ssim-1-20f521db.pth.tar" #ms-ssim
    
    # IFrameCompressor = Cheng2020Attention(N = 128)#cheng2020-attn
    # url = "https://compressai.s3.amazonaws.com/models/v1/cheng2020_attn-ms-ssim-1-c5381d91.pth.tar"  #ssim
    IFrameCompressor = IFrameCompressor.to(x_in.device)

    for p in IFrameCompressor.parameters():
        p.requires_grad = False
      
    checkpoint = load_state_dict_from_url(url, progress=True, map_location=x_in.device)
    checkpoint = load_pretrained(checkpoint)
    IFrameCompressor.load_state_dict(checkpoint)
    x_hat = IFrameCompressor(x_in)
    utils_tv.save_image(x_hat["x_hat"], os.path.join('/media/EXT0/daole/sd_scripts/VVC_Python/compressai/output/', f"test_out_bmshj2018-hyperprior_mse.png"), nrow=1)
    # string, shape = IFrameCompressor.compress(x_in)
    # x_hat = IFrameCompressor.decompress(string, shape)
    # print(x_hat["x_hat"].size())
    # x_in_lr = CNet(x_in) 
    # start_time = time.time()
    # PNG2YUV('/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/generate_images/generated_image_x0_GDP_inp_box_jpeg_input_SubSpace_TA_Enc_semantic/images_144k/000000.png', '/media/EXT0/daole/sd_scripts/VVC_Python')
    # print("--- %s seconds for png2yuv ffmpeg os ---" % (time.time() - start_time))

    # start_time = time.time()
    # PNG2YUV('/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/generate_images/generated_image_x0_GDP_inp_box_jpeg_input_SubSpace_TA_Enc_semantic/images_144k/000000.png', '/dev/shm/')
    # print("--- %s seconds for png2yuv dev ---" % (time.time() - start_time))

    # # PNG2YUV('/media/EXT0/daole/VCM_Proposed/data/test_in_COCO/000000.png')
    # # YUV2PNG('/media/EXT0/daole/sd_scripts/networks/out_256x256_1fps_8bit.yuv')
    # # YUV2PNG('/media/EXT0/daole/sd_scripts/VVC_Python/out_256x256_1.yuv')
    # # Encoder_VVC ('/media/EXT0/daole/sd_scripts/VVC_Python/input_256x256_1.yuv')
    # start_time = time.time()
    # Encoder_VVC_Single_QP49 ('/media/EXT0/daole/sd_scripts/VVC_Python/input_256x256_1.yuv', '/media/EXT0/daole/sd_scripts/VVC_Python/')
    # print("--- %s seconds for Encoder_VVC os.system---" % (time.time() - start_time))

    # start_time = time.time()
    # Encoder_VVC_Single_sp ('/media/EXT0/daole/sd_scripts/VVC_Python/input_256x256_1.yuv', '/dev/shm/')
    # print("--- %s seconds for Encoder_VVC ---" % (time.time() - start_time))
    # YUV2PNG ('/media/EXT0/daole/sd_scripts/VVC_Python/rec_QP52_256x256_1.yuv')
    # YUV2PNG('/media/EXT0/daole/sd_scripts/VVC_Python/rec_QP52_256x256_1.yuv', '/media/EXT0/daole/sd_scripts/VVC_Python/')
    # VVC_FFPMEG('/media/EXT0/daole/sd_scripts/VVC_Python/input_256x256_1.yuv', BITSTREAM_FOLDER)
    # Decoder_VVC
    # start_time = time.time()
    # img = Image.open('/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/generate_images/generated_image_x0_GDP_inp_box_jpeg_input_SubSpace_TA_Enc_semantic/images_144k/000000.png')
    # # # img_bgr = cv2.imread('/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/generate_images/generated_image_x0_GDP_inp_box_jpeg_input_SubSpace_TA_Enc_semantic/images_144k/000000.png')
    # # # img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    # # # To TENSOR
    # img = TF.to_tensor(img)[None, :]
    

    # # yuv = rgb_yuv(img)
    # # img = np.array(img)
    # # yuv = rgb_to_yuv(img)
    # # print(yuv.size)
    # # 

    # yuv = RgbToYuv420()
    # output = yuv(img)
    # print("--- %s seconds for png2yuv kornia ---" % (time.time() - start_time))

    # print(output[0].type())
    # save_yuv_420("out_test.yuv", output[0], output[1][:,0,:,:], output[1][:,1,:,:])
    # # yuv = rgb_to_yuv420 (img)
    # utils.save_image(yuv, os.path.join('/media/EXT0/daole/sd_scripts/VVC_Python/', f"input_256x256_1_py.yuv"), nrow=1)
    # Encoder_VVC_Single('/media/EXT0/daole/sd_scripts/VVC_Python/input_256x256_1.yuv', '/media/EXT0/daole/sd_scripts/VVC_Python')
if __name__ == "__main__":
    main()
