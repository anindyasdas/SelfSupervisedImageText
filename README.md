# StackGAN_v2autoencoder

The original StackGAN_v2 architecture is modified to form an autoencoder. Resnet as encoder, StackGAND_v2 as decoder
# Dataset
- Download [Flower images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
- Rename the jpg folder to images and unzip 102flowers.zip and put it inside 102flowers folder
- put 102flowers folder inside data folder
- Download [Birds data](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view) and put inside Data/
- Download [image data](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) Extract them to Data/birds/
# Running Autoencoder

Active driver prgram is main1.py
### To run on flower dataset
```
python main1.py --cfg cfg/flowers_3stages.yml --gpu 0
```
### to run on birds dataset
```
python main1.py --cfg cfg/birds_3stages.yml --gpu 0
```
Models will automatically saved after a fixed number of iteration, to restart from a failed step edit netG_version in respective .yml file
