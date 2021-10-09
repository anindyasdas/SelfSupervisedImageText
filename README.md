# Self-supervised Image-to-text and Text-to-image Synthesis(./imageae.png)

This is the official implementation of Self-supervised Image-to-text and Text-to-image Synthesis.

# Dataset
We use [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [Oxford-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) datasets in this work.
- Download [Flower images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
- Rename the jpg folder to images and unzip 102flowers.zip and put it inside 102flowers folder
- put 102flowers folder inside data folder
- Download [Birds data](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view) and put inside Data/
- Download [image data](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) Extract them to Data/birds/
# Dependencies
 - pytorch
 - torchvision
 - tensorboardX
 - pickle

# Training
## Training the image autoencoder
The driver program for training the image autoencoder is main.py
### To train the image autoencoder on flower dataset
```
python main.py --cfg cfg/flowers_3stages.yml --gpu 0
```
### To train the image autoencoder birds dataset
```
python main.py --cfg cfg/birds_3stages.yml --gpu 0
```
Models will automatically saved after a fixed number of iteration, to restart from a failed step edit netG_version in respective .yml file
## Training the text autoencoder
```
python run_text_test.py dataset_type Input_Folder output_file.txt
```
- For Flower Dataset dataset_type=1, for Birds Dataset dataset_type=2
e.g. 
```
python run_text_test.py 2 /home/user/dev/unsup/data_datasets/CUB_200_2011 outbirds_n.txt
```
## Training the mapping networks
## Train the GAN-based mapping network
```
python MappingImageText.py Dataset_folder
```
e.g.
```
python MappingImageText.py /home/user/dev/unsup/data_datasets/CUB_200_2011
```
## Train the MMD-based mapping network
```
python mmd_ganTI.py --dataset /home/das/dev/data_datasets/birds_dataset/CUB_200_2011 --gpu_device 0
```
```
python mmd_ganIT.py --dataset /home/das/dev/data_datasets/birds_dataset/CUB_200_2011 --gpu_device 0
```

