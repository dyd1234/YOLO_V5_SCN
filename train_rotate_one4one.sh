python train_rotate_One4One.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_tests_one4one_0 --fixed_angle 0 --workers 20 --imgsz 320 --optimizer Adam

python train_rotate_One4One.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_tests_one4one_30 --fixed_angle 30 --workers 20 --imgsz 320 --optimizer Adam

python train_rotate_One4One.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_tests_one4one_60 --fixed_angle 60 --workers 20 --imgsz 320 --optimizer Adam

python train_rotate_One4One.py --data stanford_dogs_new.yaml --epochs 500 --weights '' --cfg yolov5s.yaml --hyp data/hyps/hyp.scratch-low-non-rotate_adam.yaml --batch-size 128 --device 0 --cache ram --seed 50 --project YOLOv5_Rotate_tests_one4one_90 --fixed_angle 90 --workers 20 --imgsz 320 --optimizer Adam
