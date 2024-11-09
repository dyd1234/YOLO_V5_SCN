
# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 200 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 64 --device 0 --cache ram --seed 200 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 640 --optimizer Adam



# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1000 --weights '' --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1000 --weights '' --cfg yolov5s_scn_neck_head_d2.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d2 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1000 --weights '' --cfg yolov5s_scn_neck_head_d3.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d3 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1000 --weights '' --cfg yolov5s_scn_neck_head_d5.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d5 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1000 --weights '' --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d8 --workers 20 --imgsz 320 --optimizer Adam


# the following type is useful 
# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d2.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d2 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d3.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d3 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d5.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d5 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d8 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1000 --weights '' --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam


#####################
# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 200 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 2000 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d8 --workers 20 --imgsz 320 --optimizer Adam


# new train mode 10.17 These two are working for the thesis, but dont pick pretrained weights
# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam
# 
# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d8 --workers 20 --imgsz 320 --optimizer Adam


# new 10.17 working case , good to go
python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d8 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d1 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s_scn_neck_head_d16.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d16 --workers 20 --imgsz 320 --optimizer Adam

# new 10.24 working case , try 0 to 90
# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_quarter_d8 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s_scn_neck_head_d1.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_quarter_d1 --workers 20 --imgsz 320 --optimizer Adam

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s_scn_neck_head_d16.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d16 --workers 20 --imgsz 320 --optimizer Adam