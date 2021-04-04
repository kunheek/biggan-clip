#!/bin/bash
project_dir=$PWD

mkdir -p datasets/coco14
cd datasets/coco14
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip 
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
cd $project_dir