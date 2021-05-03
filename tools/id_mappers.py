"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Description:
For SYNTHIA will use imageio to read the labels instead of PIL
You may need to run
    imageio_download_bin freeimage
to download the required FreeImage lib.
"""

import numpy as np
import imageio

from PIL import Image

class SynthiaMapper(object):

    """See http://synthia-dataset.cvc.uab.es/SYNTHIA-AL/Readme.txt

        Class		R	G	B	ID
        void		0	0	0	0
        sky		70	130	180	1
        Building	70	70	70	2
        Road		128	64	128	3
        Sidewalk	244	35	232	4
        Fence		64	64	128	5
        Vegetation	107	142	35	6
        Pole		153	153	153	7
        Car		0	0	142	8
        Traffic sign	220	220	0	9
        Pedestrian	220	20	60	10
        Bicycle		119	11	32	11
        Motorcycle	0	0	230	12
        Parking-slot	250	170	160	13
        Road-work	128	64	64	14
        Traffic light	250	170	30	15
        Terrain		152	251	152	16
        Rider		255	0	0	17
        Truck		0	0	70	18
        Bus		0	60	100	19
        Train		0	80	100	20
        Wall		102	102	156	21
        Lanemarking	102	102	156	22
    """

    #
    # Synthia class names
    #
    MAP = {1: 10, # sky
           2:  2, # building
           3:  0, # road
           4:  1, # sidewalk
           5:  4, # fence
           6:  8, # veg
           7:  5, # pole
           8: 13, # car
           9:  7, # traffice sign
          10: 11, # person
          11: 18, # bicycle
          12: 17, # motorcycle
          15:  6, # traffic light
          16:  9, # terrain
          17: 12, # rider
          18: 14, # truck
          19: 15, # bus
          20: 16, # train
          21:  3} # wall

    UNLABELLED = 255

    @staticmethod
    def read(filepath):
        return np.asarray(imageio.imread(filepath, format='PNG-FI'))[:,:,0]

    @staticmethod
    def ext():
        return "*.png"

    def __getitem__(self, key):
        return SynthiaMapper.MAP[key]

class GameMapper(object):

    #
    # GTA class names
    #
    GAME_CLASSES = [
                'unlabeled','ego vehicle','rectification border','out of roi','static', \
                'dynamic','ground','road' ,'sidewalk','parking', \
                'rail track','building', 'wall','fence' ,'guard rail', \
                'bridge','tunnel','pole','polegroup','traffic light', \
                'traffic sign', 'vegetation','terrain','sky' ,'person', \
                'rider', 'car','truck','bus' ,'caravan', \
                'trailer','train', 'motorcycle', 'bicycle','license plate']

    UNLABELLED = 255

    # equivalent to CS
    MAP = {7:  0,
           8:  1,
           11: 2,
           12: 3,
           13: 4,
           17: 5,
           19: 6,
           20: 7,
           21: 8,
           22: 9,
           23: 10,
           24: 11,
           25: 12,
           26: 13,
           27: 14,
           28: 15,
           31: 16,
           32: 17,
           33: 18,
            0: 255}

    @staticmethod
    def read(filepath):
        return np.array(Image.open(filepath), dtype=np.uint32)

    @staticmethod
    def ext():
        return "*.png"

    def __getitem__(self, key):
        return GameMapper.MAP[key]

class CityscapesMapper(object):

    """
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    """

    #
    # Cityscapes mapping
    #
    MAP = { 7:   0, # 'road'         
            8:   1, # 'sidewalk'     
           11:   2, # 'building'     
           12:   3, # 'wall'         
           13:   4, # 'fence'        
           17:   5, # 'pole'         
           19:   6, # 'traffic light'
           20:   7, # 'traffic sign' 
           21:   8, # 'vegetation'   
           22:   9, # 'terrain'      
           23:  10, # 'sky'          
           24:  11, # 'person'       
           25:  12, # 'rider'        
           26:  13, # 'car'          
           27:  14, # 'truck'        
           28:  15, # 'bus'          
           31:  16, # 'train'        
           32:  17, # 'motorcycle'   
           33:  18, # 'bicycle'      
            0: 255} # 'unlabeled'    

    UNLABELLED = 255

    @staticmethod
    def read(filepath):
        return np.array(Image.open(filepath), dtype=np.uint32)

    @staticmethod
    def ext():
        return "*labelIds.png"

    def __getitem__(self, key):
        return CityscapesMapper.MAP[key]

def get_mapper(dataname):

    maps = {"cs": CityscapesMapper,
            "synthia": SynthiaMapper,
            "gta": GameMapper}

    assert dataname.lower() in maps, "Specify data from [cs|synthia|gta]"
    return maps[dataname.lower()]
