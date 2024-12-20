# YOLO_V5_SCN

For YOLO v5 related contents and how to train the models in YOLO v5 methods, Please check the YOLOv5_readme.md in this repo.

This repo provides the basic python scripts for SCN under rotation and scaling, some other modifications will be added later.

Some related shell scripts shows how I choose to train the SCN_YOLO Models with some other hyper_parameters setting. 


For models setting, Please check the file **models\common.py** to check the basic layers for both original YOLO and SCN type. For the defination of the YOLO, Please check the **models\yolo.py** file and check with **train_xx_scn.py** to see how the model is defined and be used across different files.

# implementation
## model and SCN defination
Please check the **models/common.py** file to check the layer defination for the SCN and non-SCN type. The defination of the ConV layer, which is the basic **conv+act+bn** implementation for bother SCN type and original type. Other layers are embedding the basic layers to implement.

## YOLO model and implementation
The YOLO model is defined in the yolo.py. **Please check the line 830 and 831 for the scn and non-scn YOLO defination**. You can jumo into the stuff to see how the mode defined. If you want to check the forward function, you can try to make the Please notice that SCN model define the hypernetwork forward pass in the **DetectionModel_SCN**. The result of the hypernet will be loaded into **BaseModel_SCN**. In the **BaseModel_SCN**, the layer will run the layers in a for loop and call the forward function, **Please check the **line 530** to see how the forward pass make the yolo-SCN model run with partial-SCN**. 

## training scripts
**train_rotate_scn.py** has upgraded to embed the **repair** runftions and beta2 penality.  

The beta2 and how it was transferred to the **compute_loss** class was changed a bit to fit the yolo loss functions, which means the beta2 are used by each sub loss. **Please check the from the line 106 in the utils/loss.py** to see how ComputeLoss work with the beta2 penality, mainly in the **__call__** function. 

## hyper_paramter
The hyper_parameter can be found in the **data/hyp**. The script **train_rotate_scn_head_neck_hyp_evo.py** provide the example on how to use the hyper-parameter with the hyperparameter evolution. 

