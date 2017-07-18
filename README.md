# YOLO1 Keras Model

Introduction
---------------------------------------------------------------------------------------------------

This is a Keras implementation of YOLO1 neuron network. 

    YOLO Real-Time Object Detection : YOLO paper please reference to http://pjreddie.com/darknet/yolo/


Usage
---------------------------------------------------------------------------------------
DataSets

    defined in detregion.cfg :
    
        trainset: indicate train list file
        valset: indicate validate list file
        testfile: indicate test file list for predict test
        videofile: indicat video file for predict test
        numberof_train_samples: maximum images to train, normally set to larger than trainsets to use all trainsets.
            you can set smaller for debug

Pretrained weight: ( I cannot train it very well, it is just for a example)

    https://drive.google.com/file/d/0BzLj4O82o8Hvc29qOFJpVHFaS2s/view?usp=sharing


Training and Debuging (need Python-OPENCV)

    Python main_bnum.py train [pretrained_Keras_model.h5]
    
    It will read-in all training images, so it maybe probably out of memory if trainSets is too large.
    
    in detregin.cfg setting: it will show image every batch end.
        
        (need Python-OPENCV , workon cv)
        debugimg=1
        imagefordebugtrain=imagefordebug.txt
    
    
    
Train_on_batch

    Python main_bnum.py train_on_batch [pretrained_Keras_model.h5]
    
    It will only read-in 1 batch images for each training, so there is no out of memory issue. But may take
    longer time to train because of it read image from disk for every batch.

    in detregin.cfg setting: it will show image every batch end.
        
        (need Python-OPENCV , workon cv)
        debugimg=1 
        imagefordebugtrain=imagefordebug.txt

TestOneFile

    Python main_bnum.py testoneile pretrained_Keras_model.h5 xxx.jpg 
    
    It will output predicted.png file contain bbox.

    
TestFile (need Python-OPENCV , workon cv)

    Python main_bnum.py testfile pretrained_Keras_model.h5 
    
    It will read in test images defined in detregion.cfg and show images one by one with predicted bbox on the screen.
       bbox in green-color is truth, white-colors is prediction 
    predicted bbox show only "confidence value" > thresh 
    
    at image top : predict bbox IOU value
    at predict box top (white color) : class probability
    at predict box bottom (lightblue color) : confidence value
    

TestVideo (need Python-OPENCV , workon cv)

    Python main_bnum.py testvideo pretrained_Keras_model.h5 
    
    It will read video file defined in detregion.cfg and show video with predicted bbox on the screen.
    predicted bbox show only "confidence value" > thresh (default 0.6)
    
    at image top : predict bbox IOU value
    at predict box top (white color) : class probability
    at predict box bottom (lightblue color) : confidence value
    
Testvideosocket (need Python-OPENCV , workon cv)

    Python main_bnum.py testvideosocket pretrained_Keras_model.h5 
    -- it will wait for imageClient.py to send images/video --
    
    python imageClient.py imagefilelist.txt (must be .txt) -- it will send images for prediction 
    
    python imageClient.py xxxxx.mp4 (need Python-OPENCV , workon cv) -- it will send video for predcition
    
    Provide another method to test video/testimages, it act as image receiver, use imageClient.py to
        connect to it and provide the test video file or image files. 
        
Configure netwrok: 

    the network is built in builtinModel.py code, it contains yolotiny, yolosmall, yolo, vgg16 network.
    it use detregion network place on last stage (add_regionDetect in builtinModel.py code)
    the last stage size is defined in detregion.cfg (= ((classes+5)*bnum)*side^2 )
    
    network code in main_bnum.py : change "builtinModel.yolotiny_model((448, 448, 3))" to yours.
    (448, 448, 3) is input image size with 3 color
    ------------------------------------------------------------------------------------------------------
    model = builtinModel.add_regionDetect(builtinModel.yolotiny_model((448, 448, 3)), (cfgconst.side**2)*(cfgconst.classes+5)*cfgconst.bnum)
    ------------------------------------------------------------------------------------------------------
    
Code explanation
---------------------------------------------------------------------------------------------

    main_bnum.py: entry point
    
    genregiontruth_bnum.py: prepare truth and train data
    
    detregionloss_bnum.py: loss function
    
    builtinModel.py: define object detection neueon network structure
    
    imagefordebug.txt: image for train debug use case
    
    detregion.cfg: parameter setting file
        
   


How to Training
------------------------------------------------------------------------------------------------

    1. Learning rate start from 0.1, if loss doesnt decrease for 3 epochs, learning rate decrease by 1/2.
        if loss doesnt decrease for 10*3 epochs, then stop training
        
    2. Trainging image is generated by randomly shifting, croping, fliping, contrasting, brighting dataset image.
        
    3. add COCO dataset to trainSets
    

Running enviroment
--------------------------------------------------------------------------------------------

Tools :

    Ubuntu 16.04 LTS 64-bit, Python 2.7.12
  
    Keras with tensorflow backend : Neoron Network training

    Python-OpenCV : display image and video on screen

DataSets :
  
    VOC : detail please reference http://pjreddie.com/darknet/yolo/ to create VOC datasets, 
    here are instructions from website.
    
    Get The Pascal VOC Data: (it may take long time to download)
    in VOCdevkit/ dir
    -------------------------------------------------------------------------------
    curl -O https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    curl -O https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar
    -------------------------------------------------------------------------------
    
    Generate Labels for VOC:
    in VOCdevkit/VOC2007/labels/ and VOCdevkit/VOC2012/labels/
    -------------------------------------------------------------------------------
    <object-class> <x> <y> <width> <height>  --- label format
    -------------------------------------------------------------------------------
    curl -O https://pjreddie.com/media/files/voc_label.py
    python voc_label.py
    -------------------------------------------------------------------------------
    
    
    COCO : run coco_dataset_Creator.py, you may need to modify code to fit you folder path. It may take several hours 
            to download all images.

Directory structure :

    VGG/ : this git
    
    VOC/voc_label.py : generate VOC lables, you can get this file from http://pjreddie.com/darknet/yolo/
    
    VOC/VOCdevkit : VOC datasets
    
    COCO/annotations : COCO datasets (2014 train/val object instance, 2014 train/val person keypoints), you can get from
        http://mscoco.org/dataset/#download
    
    COCO/coco : MS COCO API, you can get from https://github.com/pdollar/coco
