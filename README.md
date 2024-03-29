# 3DMVGOD
Weakly supervised 3D object detection using multi-view geometry
1. Generate pseudo 3D bbox based on multi-view geometry consistency
2. Train existing supervised 3D object detector using pseudo 3D annotations 


## Requirements

Build anaconda environment with all the required packages

   ```
   conda env create -f env_3dod.yml
   ```


## Dataset
For [ScanNet](http://www.scan-net.org/changelog#scannet-v2-2018-06-11), we download version2 from official website 
(https://http://www.scan-net.org/). 

For [SUNRGBD](http://rgbd.cs.princeton.edu/data/SUNRGBD.zip), we download from official website 
(http://rgbd.cs.princeton.edu/challenge.html). 
   
   
## Usage 

### Data Preparation for ScanNet dataset
First we extract all the RGB-D frames (color and depth frames), camera intrinsics and extrinsics files from 
compressed binary file (<scene_id>.sens) by running ```./data/scannet/SensReader/bash_reader.py``` 
This step will generate four subfolders (i.e., 'color', 'depth', 'pose', 'intrinsic') in each scan folder.

Then we extract valid frames from each scene by checking the camera extrinsic files and instance segmentation
 annotations. 
Simply run  ```./data/scannet/process_dataset.py --min_frames 5 --min_angle 5. --min_ratio 15```
This step will generate two subfolders (i.e., 'instance-filt', 'bbox2d_*class') and one file (i.e., 
'<sceneid>_validframes_*class.txt') in each scan folder. Also, it will generate two files (i.e., 
'log_process_data.txt', 'sceneid_valid.txt') in the root data dir..


### Train mult-label classifier to extract CAM
We use the ResNet-50 backbone as our 2D feature extractor.

For training, run ```./CAM/train_cam.py --modelname ResNet50 --pretrained True --input_size 224 --batch_size 64 --max_epoch 20 
--learning_rate 0.0001 --weight_decay 0.00001 --decay_step 5 --decay_ratio 0.7 ```.
If 'pretrained' is True, the model will be first  pre-trained on ImageNet dataset, otherwise it will be trained 
from scratch. 
Note that if pretrained is not performed, the input arguments should be tuned accordingly, e.g., 
```--learning_rate 0.001 --max_epoch 100 --weight_decay 0.0001 --decay_step 20 --decay_ratio 0.7```

For extracting CAM, run ```./CAM/extract_cam.py  --checkpoint $CHECKPOINTPATH --modelname ResNet50 --to_save True```
This step generates CAM results (7x10xnum_class) as numpy file for each frame in the training dataset. 
Note that for visualizing CAM results, make ```--to_visualize True```, it will generate image file (with heatmap) for 
all the target classes of each frame in the training dataset.


### Pseudo 3D Bbox annotation generation
run ```./Box3dGenerator/generator.py ``` 
