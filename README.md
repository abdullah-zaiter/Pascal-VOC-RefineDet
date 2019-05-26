# Pascal-VOC-RefineDet

A [PyTorch](http://pytorch.org/) - version 1.1 implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897 ). The official and original Caffe code can be found [here](https://github.com/sfzhang15/RefineDet).

<p align="right">
  <img  src="https://github.com/abdullah-zaiter/Pascal-VOC-RefineDet/blob/master/doc/ssd.png">
</p>


### Table of Contents
- <a href='#performance'>Performance</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-refinedet'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#realtime-testing'>Realtime Testing</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Performance

#### VOC2007 Test

##### mAP (*Single Scale Test*)

| Arch | result |
|:-:|:-:|
| RefineDet320 | 71.81% |
| RefineDet512 | Not trained locally yet|

## Installation
The conda environment used in this project can be imported from the file environment.yml, or you can download the packages manually:
- Install [PyTorch](http://pytorch.org/) version 1.1
- Install [OpenCV](https://opencv.org/) at least version 3
- Clone this repository.
  * Note: Only Python 3+ supported.
- Then download the dataset by following the [instructions](#datasets) below.

#### For real-time loss visualization during training
- Install [Visdom](https://github.com/facebookresearch/visdom) 
  * To use Visdom in the browser:
  ```Shell
    # First install Python server and client
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).

#### For real-time Webcam evaluation after trainging:
- Install [Qtcreator](https://www.qt.io/qt-features-libraries-apis-tools-and-ide/) 
```Shell
    sudo apt install qtcreator
```


## Datasets
To make things easy, a bash script was provided to handle the dataset downloads and setup.  Also a simple dataset loaders that inherit  `torch.utils.data.Dataset` was provided, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).


### VOC Dataset
PASCAL VOC: Visual Object Classes

### OBS: ALL SCRIPTS MUST BE RUN FROM INSIDE SRC DIRECTORY

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ./database/
sh data/scripts/VOC2007.sh # <directory>
```

## Training RefineDet
- Download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `src/weights` dir:

```Shell
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RefineDet320 or RefineDet512 using the train scripts `train_refinedet320.sh` and `train_refinedet512.sh`. You can manually change them as you want.

```Shell
sh ./mainscripts/train_refinedet320.sh  #./train_refinedet512.sh
```

- Notes:
  * For training, an NVIDIA GPU is strongly recommended for speed (SPENT 20 HOURS TRAINING ON 1070 GPU)
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train_refinedet.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
sh ./mainscripts/eval_refinedet.sh --trained_model ./weights/RefineDet320/RefineDet320_VOC_final.pth
```
You can specify the parameters listed in the `eval_refinedet.py` file by flagging them or manually changing them.  
## Realtime Testing
For realtime testing using the webcam:

```Shell
sh ./mainscripts/real_time_evaluation.sh
```
### A result of webcam realtime test:
<p align="center">
  <img  src="https://github.com/abdullah-zaiter/Pascal-VOC-RefineDet/blob/master/doc/real_time_identification.png">
</p>

## References
- [Original Implementation (CAFFE)](https://github.com/sfzhang15/RefineDet) - the paper which the implmentation was based on
- [luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch) - Adapted to the last pytorch 1.1, fixed some bugs and added the real time testing functionality
