
# # d8 head_neck_scaling
# python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d8_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d8.yaml --seed 50 --workers 12 --optimizer Adam

# # d1 head_neck_scaling
# # python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d1_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d1.yaml --seed 50 --workers 12 --optimizer Adam
# # python train_scaling_scn.py --batch 64 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d1_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d1.yaml --seed 50 --workers 12 --optimizer Adam

# # d2 head_neck_scaling
# # python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 2000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d2_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d2.yaml --seed 50 --workers 12 --optimizer Adam


# # d3 head_neck_scaling
# # python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d3_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d3.yaml --seed 50 --workers 12 --optimizer Adam


# # d5 head_neck_scaling
# # python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d5_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d5.yaml --seed 50 --workers 12 --optimizer Adam


# no pretrained weights but with penality 
# d8 head_neck_scaling 
# the loss vanished and it is not totally ok to make the result do again
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d8_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d8.yaml --seed 50 --workers 12 --optimizer Adam
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 50 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d8_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d8.yaml --seed 50 --workers 12 --optimizer Adam

# d1 head_neck_scaling
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d1_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d1.yaml --seed 50 --workers 12 --optimizer Adam
# python train_scaling_scn.py --batch 64 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d1_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d1.yaml --seed 50 --workers 12 --optimizer Adam

# d2 head_neck_scaling
# python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 2000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d2_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d2.yaml --seed 50 --workers 12 --optimizer Adam
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 200 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d2_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d2.yaml --seed 50 --workers 12 --optimizer Adam

# d3 head_neck_scaling
# python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d3_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d3.yaml --seed 50 --workers 12 --optimizer Adam


# d5 head_neck_scaling
# python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 1000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d5_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d5.yaml --seed 50 --workers 12 --optimizer Adam

# 10.18 try

# d2 head_neck_scaling
# python train_scaling_scn.py --batch 128 --weights yolov5s.pt --data stanford_dogs_new.yaml --epochs 2000 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d2_scaling_neck_head_adam --cfg yolov5s_scn_neck_head_d2.yaml --seed 50 --workers 12 --optimizer Adam
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 500 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d2_scaling_neck_head_adam_nocos --cfg yolov5s_scn_neck_head_d2.yaml --seed 50 --workers 12 --optimizer Adam

# later on try with d1 neck_head_scaling
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 500 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d1_scaling_neck_head_adam_nocos --cfg yolov5s_scn_neck_head_d1.yaml --seed 50 --workers 12 --optimizer Adam
# python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 500 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_scaling_translation_adam.yaml --project standford_dog_scn_d8_scaling_neck_head_adam_nocos --cfg yolov5s_scn_neck_head_d8.yaml --seed 50 --workers 12 --optimizer Adam

# 10.21 tests with beta2
python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 500 --device 0 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_adam.yaml --project standford_dog_scn_d1_scaling_neck_head_adam_cos --cfg yolov5s_scn_neck_head_d1.yaml --seed 50 --workers 20 --optimizer Adam
python train_scaling_scn.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 500 --device 0 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_adam.yaml --project standford_dog_scn_d8_scaling_neck_head_adam_cos --cfg yolov5s_scn_neck_head_d8.yaml --seed 50 --workers 20 --optimizer Adam

python train_scaling_One4All.py --batch 128 --weights '' --data stanford_dogs_new.yaml --epochs 500 --device 0 --cache ram --img 320 --hyp hyp.scratch-low-non-rotate_adam.yaml --project standford_dog_one4all_scaling_adam --cfg yolov5s.yaml --seed 50 --workers 20 --optimizer Adam