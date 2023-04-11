# TopTrack: Tracking Objects By Their Top 
Multi-object tracking using the top as a keypoint for detection



## Abstract 

In recent years, the joint detection-and-tracking paradigm has been a very popular way of tackling the multi-object tracking (MOT) task. Many of the methods following this paradigm use the object center keypoint for detection. However, we argue that the center point is not optimal since it is often not visible in crowded scenarios, which results in many missed detections when the objects are partially occluded. We propose \textit{TopTrack}, a joint detection-and-tracking method that uses the top of the object as a keypoint for detection instead of the center because it is more often visible. Furthermore, \textit{TopTrack} processes consecutive frames in separate streams in order to facilitate training. We performed experiments to show that using the object top as a keypoint for detection can reduce the amount of missed detections, which in turn leads to more complete trajectories and less lost trajectories. \textit{TopTrack} manages to achieve competitive results with other state-of-the-art trackers on two MOT benchmarks.

## Main results

### Results on MOT test set

| Dataset    |  MOTA | HOTA | IDF1 | IDS | MT | ML |
|--------------|-----------|-----------|--------|-------|----------|----------|
|MOT17       | 64.8 |  47.9 | 58.2 | 10083 | 38.7% | 11.5% |
|MOT20       | 46.3 | 26.8 | 27.6 | 23227 | 20.0% | 22.0% |

Results are obtained using private detections.

## Installation
* Install dependencies. We use python 3.8 and pytorch 1.11.0
```
conda create -n TopTrack python=3.8
conda activate TopTrack
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
* Install [COCOAPI](https://github.com/cocodataset/cocoapi):

~~~
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
~~~


## Data preparation

### MOT17
The data processing is performed by our script 

~~~
cd $TopTrack_ROOT/tools/
bash get_mot_17.sh 
~~~

This script:
* Downloads and extracts the dataset
* Converts the annotations into COCO format
* Creates the validation set by splitting the training set into two halves ```train_half``` and ```val_half```

The final structure should follow:
~~~
${TopTrack_ROOT}
|-- data
└-- |-- mot17
      └-- |--- train
          |   |--- MOT17-02-FRCNN
          |   |    |--- img1
          |   |    |--- gt
          |   |    |   |--- gt.txt
          |   |    |   |--- gt_train_half.txt
          |   |    |   |--- gt_val_half.txt
          |   |    |--- det
          |   |    |   |--- det.txt
          |   |    |   |--- det_train_half.txt
          |   |    |   |--- det_val_half.txt
          |   └--- ...
          |--- test
          |   |--- MOT17-01-FRCNN
          |   └--- ...
          └---| annotations
              |--- train_half.json
              |--- val_half.json
              |--- train.json
              └--- test.json
~~~
  
### MOT20 
The data processing is performed by our script 

~~~
cd $TopTrack_ROOT/tools/
bash get_mot_20.sh 
~~~

This script:
* Downloads and extracts the dataset
* Converts the annotations into COCO format
* Creates the validation set by splitting the training set into two halves ```train_half``` and ```val_half```

The final structure should follow:
~~~
${TopTrack_ROOT}
|-- data
└-- |-- mot20
  └-- |--- train
      |   |--- MOT20-01
      |   |    |--- img1
      |   |    |--- gt
      |   |    |   |--- gt.txt
      |   |    |   |--- gt_train_half.txt
      |   |    |   |--- gt_val_half.txt
      |   |    |--- det
      |   |    |   |--- det.txt
      |   |    |   |--- det_train_half.txt
      |   |    |   |--- det_val_half.txt
      |   └--- ...
      |--- test
      |   |--- MOT20-04
      |   └--- ...
      └---| annotations
          |--- train_half.json
          |--- val_half.json
          |--- train.json
          └--- test.json
~~~

### CrowdHuman
You can download the CrowdHuman dataset from their official [website](https://www.crowdhuman.org). 
Only the training set is used for pretraining and come in three folders that need to be merged after extraction

The final structure should follow:

~~~
${CenterTrack_ROOT}
|-- data
└-- |-- crowdhuman
         |-- CrowdHuman_train
         |   └-- Images
         └-- annotation_train.odgt
~~~

### References
Please cite the corresponding References if you use the datasets.

~~~
  @article{MOT16,
    title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
    shorttitle = {MOT16},
    url = {http://arxiv.org/abs/1603.00831},
    journal = {arXiv:1603.00831 [cs]},
    author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
    month = mar,
    year = {2016},
    note = {arXiv: 1603.00831},
    keywords = {Computer Science - Computer Vision and Pattern Recognition}
  }
  
  @misc{dendorfer2020mot20,
      title={MOT20: A benchmark for multi object tracking in crowded scenes}, 
      author={Patrick Dendorfer and Hamid Rezatofighi and Anton Milan and Javen Shi and Daniel Cremers and Ian Reid and Stefan Roth and Konrad Schindler and Laura   Leal-Taixé},
      year={2020},
      eprint={2003.09003},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

  @article{shao2018crowdhuman,
    title={Crowdhuman: A benchmark for detecting human in a crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv:1805.00123},
    year={2018}
  }
~~~

## Model Zoo

You can download our trained MOT17 model at:  and our trained MOT20 model at:


## Training and Evaluation
To run a demo of TopTrack, first download a model for our Model Zoo and then use the following command:

```
python demo.py ctdet --tracking --load_model $MODEL_PATH$ --demo $DATA_PATH$ --dataset $DATASET_NAME$ --dataset_version $DATASET_VERSION$ --debug 1 --pre_hm  --show_track_color
```

To train TopTrack from scratch, use the following command:
```
python main.py ctdet --tracking --exp_id $MODEL_NAME$ --dataset $DATASET_NAME$ --dataset_version $DATASET_VERSION$ --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --batch_size $BATCH_SIZE$ --num_epochs $NUM_EPOCHS$
```
Other flags can be added, see opt.py for a full list

To test TopTrack, use the following command:
```
python test.py ctdet --tracking --exp_id $MODEL_NAME$ --pre_hm --dataset $DATASET_NAME$ --dataset_version $DATASET_VERSION$ --resume
```

## License

TopTrack is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch) (image transform, resnet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).

