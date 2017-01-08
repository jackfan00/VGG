# Object Detection Neuron Network

Introduction
---------------------------------------------------------------------------------------------------

This is a Keras implementation of Object Detection Neuron Network. 

    YOLO Real-Time Object Detection : YOLO paper please reference to http://pjreddie.com/darknet/yolo/


Usage
---------------------------------------------------------------------------------------

Training and Debuging (need Python-OPENCV)

    Python main.py train trainlist.txt [numberOfSmaples] [pretrained_Keras_model.h5] [Debug]
    
    It will read-in all training images, so it maybe probably out of memory if trainSets is too large.
    
    "numberOfSmaples" can be specified to avoid this kind of problem. If not specify "numberOfSmaples"
    or "numberOfSmaples" greater than trainsets, it read-in all trainsets.
    
    Option "Debug" equal 1 will show image (which specified by numberOfSmaples.txt) with predicted bbox 
    on the screen when training, it can help to give a feeling about training process.
    
Train_on_batch

    Python main.py train_on_batch trainlist.txt [numberOfSmaples] [pretrained_Keras_model.h5]
    
    It will only read-in 1 batch images for each training, so there is no out of memory issue. But may take
    longer time to train because of it read image from disk for every batch.
    
TestFile (need Python-OPENCV)

    Python main.py testfile testlist.txt thresh pretrained_Keras_model.h5
    
    It will show images with predicted bbox on the screen

TestVideo (need Python-OPENCV)

    Python main.py testvideo videofile thresh pretrained_Keras_model.h5
    
    It will show video with predicted bbox on the screen
    
    
Code explanation
---------------------------------------------------------------------------------------------

    main.py: entry point
    
    genregiontruth.py: prepare truth and train data
    
    detregionloss.py: loss function
    
    builtinModel.py: define object detection neueon network structure
    
    imagefordebug.txt: image for train debug use case
    
    detregion.cfg: detection parameter



Running enviroment
--------------------------------------------------------------------------------------------
Tools :
  
    Keras with tensorflow backend : Neoron Network training

    Python-OpenCV : display image and video on screen

DataSets :
  
    please reference http://pjreddie.com/darknet/yolo/ to create VOC datasets
    
