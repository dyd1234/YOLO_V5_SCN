# YOLO_V5_SCN

For YOLO v5 related contents and how to train the models in YOLO v5 methods, Please check the YOLOv5_readme.md in this repo.

This repo provides the basic python scripts for SCN under rotation and scaling, some other modifications will be added later.

Some related shell scripts shows how I choose to train the SCN_YOLO Models with some other hyper_parameters setting. 


For models setting, Please check the file **models\common.py** to check the basic layers for both original YOLO and SCN type. For the defination of the YOLO, Please check the **models\yolo.py** file and check with **train_xx_scn.py** to see how the model is defined and be used across different files.


