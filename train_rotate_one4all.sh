
# conda activate scn_new_910

# python train_rotate_One4All.py --data stanford_dogs_new.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 150 --project YOLOv5_Rotate_tests_one4all --workers 20 --imgsz 320 --optimizer Adam

# sample stuff
# python train_rotate_One4All.py --data stanford_dogs_new.yaml --epochs 2000 --weights yolov5s.pt --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_One4All_adam --workers 20 --imgsz 320 --optimizer Adam

# with the backbone freeze
# python train_rotate_One4All.py --data stanford_dogs_new.yaml --epochs 2000 --weights yolov5s.pt --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_One4All_adam --workers 20 --imgsz 320 --optimizer Adam --freeze 10

# try with another 640 size ， better try later
# python train_rotate_One4All.py --data stanford_dogs_new.yaml --epochs 2000 --weights yolov5s.pt --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_One4All_adam --workers 20 --imgsz 640 --optimizer Adam --freeze 10

# python train_rotate_scn.py --data stanford_dogs_new.yaml --epochs 2000 --weights yolov5s.pt --cfg yolov5s_scn_neck_head_d8.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 256 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_scn_neck_head_test_d8 --workers 20 --imgsz 320 --optimizer Adam --freeze 10

# align to the SCN setting for head and neck 

# 10.18 try with the same settings with the SCNs and see how it is going to work
python train_rotate_One4All.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_tests_one4all --workers 20 --imgsz 320 --optimizer Adam

