# BigGAN + CLIP
This project is inspired by the [ContraGAN](https://arxiv.org/abs/2006.12681), [OpenAI CLIP](https://openai.com/blog/clip/) and other text-to-image GAN papers.  
<p align="center">
  <img width="285" height="260" src="https://github.com/kunhee-kim/biggan-clip/blob/master/assets/discriminator.png">
</p>

### Remark
**This is not a research project, but just a hobby project.** It means that the quality of the generated images are far from the state-of-the-art.


---
## Installation
First, you need to install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html).  

Then, you can install the environment following below instructions.
```sh
git clone https://github.com/kunhee-kim/biggan-clip
cd biggan-clip
conda env create -f conda_env.yml
```

### Datasets Preparation
It's simple as below!
```sh
sh download.sh
```
You also need to downloaded precalculated inception statistics from [DM-GAN repo](https://github.com/MinfengZhu/DM-GAN) and place it to *./datasets/coco14/coco14_val_npz*.

---
### Training
```sh
conda activate biggan-clip
python train.py --dataset coco14 --image_size 64 --exp_name exp0 --D_SN
```
This will create *expr/exp0* folder and logs the training loss and sample images in to the folder.

---
## Results
<p align="center">
  <img width="730" height="432" src="https://github.com/kunhee-kim/biggan-clip/blob/master/assets/sample1.png">
</p>
<p align="center">
  <img width="730" height="432" src="https://github.com/kunhee-kim/biggan-clip/blob/master/assets/sample2.png">
</p>
