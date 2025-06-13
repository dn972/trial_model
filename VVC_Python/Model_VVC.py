import re

class SeqInfo(object):
    def __init__(self, yuvFileName: str, numFrames: int, width=None, height=None, frameRate=None):
        self.yuvFileName = yuvFileName
        self.numFrames = numFrames

        if width is None:
            self.parseInfo(yuvFileName)
        else:
            self.width = width
            self.height = height
            self.frameRate = frameRate

    def parseInfo(self, yuvFileName:str):
        items = re.split("_|x",yuvFileName)
        self.width = items[1]
        self.height = items[2]
        self.frameRate = items[3]
    
        
class CommonCfg(object):
    QP_BASE_LIST = [22, 27, 32, 37, 42, 47]
    # QP_BASE_LIST = [32, 37, 42, 47, 52, 57]#BG
    # QP_BASE_LIST = [52,57]
    INTRA_PERIOD = -1

    LIST_SEQUENCE_SDR = (
        # SeqInfo("BQSquare_416x240_60", numFrames = 600),
        # SeqInfo("BasketballPass_416x240_50", numFrames = 500),
        # SeqInfo("BlowingBubbles_416x240_50", numFrames = 500),
        # SeqInfo("RaceHorses_416x240_30", numFrames = 300),
        
        # SeqInfo("BQSquareROI_416x240_60", numFrames = 600),
        # SeqInfo("BasketballPassROI_416x240_50", numFrames = 500),
        # SeqInfo("BlowingBubblesROI_416x240_50", numFrames = 500),
        # SeqInfo("RaceHorsesROI_416x240_30", numFrames = 300),

        # SeqInfo("BQSquareBG_416x240_60", numFrames = 600),
        # SeqInfo("BasketballPassBG_416x240_50", numFrames = 500),
        # SeqInfo("BlowingBubblesBG_416x240_50", numFrames = 500),
        # SeqInfo("RaceHorsesBG_416x240_30", numFrames = 300),

        SeqInfo("TVD125I_1920x1080_30", numFrames = 1),
        SeqInfo("TVD125ROI_1920x1080_30", numFrames = 1),
        # SeqInfo("TVD125BG_1920x1080_30", numFrames = 1),


    )
    
    
