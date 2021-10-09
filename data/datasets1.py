import os
from torchvision import transforms as tt
from skimage import io
from torch.utils.data import Dataset
#from text import TextLoader
from matplotlib.pyplot import imread
from torch.nn.modules.padding import ConstantPad2d
import csv
import random
from PIL import Image
from data.vocab import ShapesVocabBuilder, BirdsVocabBuilder, BillionVocabBuilder, FlowersVocabBuilder
from config import cfg
import pandas as pd
import numpy as np
import glob
import re
import pickle
import ntpath

class Image_Caption_Dataset(Dataset):
    def __init__(self, vocab_builder):
        self.vocab_builder = vocab_builder

    def __len__(self):
        raise NotImplementedError("This method must be implemented in subclass")

    def __getitem__(self, idx):
        raise NotImplementedError("This method must be implemented in subclass")

    def build_vocab(self, listof_files):
        self.vocab_builder.load_texts(listof_files)

    def get_vocab_builder(self):
        return self.vocab_builder


class ShapesDataset(Image_Caption_Dataset):
    def __init__(self, rootdir, vocab_builder=ShapesVocabBuilder(), split='train', transform=None):
        """
        Initialize shapes dataset
        :param rootdir: root directory
        :param split: one of {train, val}
        :param transform: transformation for the images
        """
        super(ShapesDataset, self).__init__(vocab_builder)
        self.transform = transform
        self.rootdir = rootdir
        self.split_dir = os.path.join(rootdir, split)
        self.img_dir = os.path.join(self.split_dir, 'images')
        self.image_caption_csv = os.path.join(self.split_dir, 'image_captions.csv')

        # build vocab from train and val set together
        train_captions_path = self.image_caption_csv
        val_captions_path = self.image_caption_csv.replace('train', 'val')
        self.build_vocab([train_captions_path, val_captions_path])

        # Note:
        # We read the captions twice from the file(s). Once for storing the captions in a list (self.image_captions)
        # which is accessed in __get_item__ and once for building the vocabulary. The same happens in the birds dataset.
        # For now it's not really an issue but maybe when datasets become larger.
        with open(self.image_caption_csv) as csvfile:
            reader = csv.reader(csvfile)
            self.image_captions = [row for row in reader][1:]  # skip header
        sentences = [row[1] for row in self.image_captions]

        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in sentences])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_captions[index][0] + '.png')
        img = io.imread(img_path)
        caption = self.image_captions[index][1]

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()  # remove batch size dim since dataloader will add it
        caption_length = caption_length.squeeze()

        untransformed_img = tt.ToTensor()(img)
        if self.transform:
            img = self.transform(img)

        return img, untransformed_img, caption_encoded, caption_length


class BirdsDataset(Image_Caption_Dataset):
    def __init__(self, rootdir, transform, vocab_builder=BirdsVocabBuilder(), split='train', img_format=".jpg"):
        super(BirdsDataset, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")
       
        self.train_test_split = []
        with open(os.path.join(rootdir, "train_test_split.txt")) as f:
            lines = f.readlines()
            self.train_test_split = [line.split()[1] for line in lines]

        self.allimages = []
        with open(os.path.join(rootdir, "images.txt")) as f:
            lines = f.readlines()
            self.allimages = [line.split()[1] for line in lines]

        # Sanity check
        assert len(self.allimages) == len(self.train_test_split)
        
        ##########################BBBOX ##########################
        ###########################################################
        bbox_path = os.path.join(rootdir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        self.filename_bbox = {img_file: [] for img_file in self.allimages}
        numImgs = len(self.allimages)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = self.allimages[i]
            self.filename_bbox[key] = bbox
        ###############################################################
        ###############################################################

        self.images = []

        split_identifier = self.split_ids[split]
        for idx, image in enumerate(self.allimages):
            if self.train_test_split[idx] == split_identifier:
                self.images.append(image)
         ###################################################for 80 :20 split##########
         ##########################################################################
        self.split_dir = os.path.join(os.path.normpath(rootdir + os.sep + os.pardir), split)
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        self.images = []
        if len(self.allimages) > len(filenames):
            for image in  filenames:
                self.images.append(image + '.jpg') 
        
        #####################################################################
        #####################################################################
        

        # read captions to determine max sentence length
        caption_dir = os.path.join(rootdir, "text_c10")
        captionpaths = []
        for subdir in os.listdir(caption_dir):
            for file in os.listdir(os.path.join(caption_dir, subdir)):
                captionpaths.append(os.path.join(caption_dir, subdir, file))

        # Note:
        # see Note in ShapesDataset
        captions = []
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    captions.append(lines[0].strip()) # select always the first description as caption for now
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in captions])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos

        # build vocabulary
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.images)
    
    def crop_image(self, img, box):
        #######################cropping important part################
        width, height = img.size
        r = int(np.maximum(box[2], box[3]) * 0.75)
        center_x = int((2 * box[0] + box[2]) / 2)
        center_y = int((2 * box[1] + box[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        return img
        ######################################################################

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_rootdir, self.images[idx])
        caption_path = self.__change_fileending(img_path)
        img_key = self.images[idx]
        bbox = self.filename_bbox[img_key]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB') #output is PIL
            img = self.crop_image(img, bbox)
            img = tt.Resize((128,128))(img) #Resize into 64*64
            img = tt.ToTensor()(img)
            img_padded = self.transform(img)

        caption = ""
        with open(caption_path) as f:
            captions = f.readlines()
        if len(captions) > 0:
            caption = captions[0].strip() # select always the first description as caption for now

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return img_padded, img, caption_encoded, caption_length

    def __change_fileending(self, path):
        """
        Changes fileending from image to text format
        :return: path with .txt ending instead of image format ending
        """
        if self.image_format in path:
            return path.replace(self.image_format, ".txt").replace("images", "text_c10")
        else:
            raise ValueError("Unexpected image format")
 

class BirdstextDataset(Image_Caption_Dataset):
    #Each image has multiple captions per image, It returns all the captions for birds dataset
    #Created to increase train datasize
    def __init__(self, rootdir, transform, vocab_builder=BirdsVocabBuilder(), split='train', img_format=".jpg"):
        super(BirdstextDataset, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")

        self.train_test_split = []
        with open(os.path.join(rootdir, "train_test_split.txt")) as f:
            lines = f.readlines()
            self.train_test_split = [line.split()[1] for line in lines]

        self.allimages = []
        with open(os.path.join(rootdir, "images.txt")) as f:
            lines = f.readlines()
            self.allimages = [line.split()[1] for line in lines]

        # Sanity check
        assert len(self.allimages) == len(self.train_test_split)
        
       

        self.images = []

        split_identifier = self.split_ids[split]
        for idx, image in enumerate(self.allimages):
            if self.train_test_split[idx] == split_identifier:
                self.images.append(image)
                
         ###################################################for 80 :20 split##########
         ########################################################################
        self.split_dir = os.path.join(os.path.normpath(rootdir + os.sep + os.pardir), split)
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        self.images = []
        if len(self.allimages) > len(filenames):
            for image in  filenames:
                self.images.append(image + '.jpg') 
        
        #####################################################################
        ###################################################################

        # read captions to determine max sentence length
        caption_dir = os.path.join(rootdir, "text_c10")
        captionpaths = []
        for subdir in os.listdir(caption_dir):
            for file in os.listdir(os.path.join(caption_dir, subdir)):
                captionpaths.append(os.path.join(caption_dir, subdir, file))

        # Note:
        # see Note in ShapesDataset
        self.captions = []
        for img_subpath in self.images:
            img_path = os.path.join(self.image_rootdir, img_subpath)
            caption_path = self.__change_fileending(img_path)
            f = open(caption_path , 'r', encoding='utf-8')
            for line in f:
                line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                self.captions.append(line.strip().lower())
            f.close()
        
        allcaptions =[]
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    for line in lines:
                        line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                        allcaptions.append(line.strip()) # select always the first description as caption for now
                        
                        
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in allcaptions])
        
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos
        
        
        #print('the maximum sentence length',self.max_sent_length)
        #print('number of images:', len(self.images), 'number of captions:',len(self.captions))
        # build vocabulary
        print('number of captions in birds dataset', len(self.captions))
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.captions)
    
   
    def __getitem__(self, idx):
        caption = self.captions[idx]
        _ =''

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return _, _, caption_encoded, caption_length

    def __change_fileending(self, path):
        """
        Changes fileending from image to text format
        :return: path with .txt ending instead of image format ending
        """
        if self.image_format in path:
            return path.replace(self.image_format, ".txt").replace("images", "text_c10")
        else:
            raise ValueError("Unexpected image format")
            
            
class BillionDataset(Image_Caption_Dataset):
    def __init__(self, rootdir, vocab_builder=BillionVocabBuilder(), split='train'):
        super(BillionDataset, self).__init__(vocab_builder)
        self.captiondir = rootdir
        
        captionpaths = []
        self.captions=[]
        for captionpath in glob.glob(self.captiondir):
            if len(self.captions) < 1000000: #taking only one million records
                captionpaths.append(captionpath)
                f = open(captionpath , 'r', encoding='utf-8')
                for line in f:
                    if len(self.captions) < 1000000:
                        line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                        if (len(self.vocab_builder.tokenizer.tokenize(line)) +2)  <= 75 : #with length less than the length of birds dataset
                            self.captions.append(line.strip().lower())
                f.close()
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in self.captions])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos
        # build vocabulary
        print('number of captions in billion dataset', len(self.captions))
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.captions)
    
    

    def __getitem__(self, idx):
        _ =''
        
        caption = self.captions[idx]

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return _, _, caption_encoded, caption_length

   
class BirdsDataset1(Image_Caption_Dataset):
    def __init__(self, rootdir, transform, vocab_builder=BirdsVocabBuilder(), split='train', img_format=".jpg", num_dis =3, img_size =64):
        super(BirdsDataset1, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")
        self.num_dis = num_dis
        self.size = img_size
        self.norm2 = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.train_test_split = []
        with open(os.path.join(rootdir, "train_test_split.txt")) as f:
            lines = f.readlines()
            self.train_test_split = [line.split()[1] for line in lines]

        self.allimages = []
        with open(os.path.join(rootdir, "images.txt")) as f:
            lines = f.readlines()
            self.allimages = [line.split()[1] for line in lines]

        # Sanity check
        assert len(self.allimages) == len(self.train_test_split)
        
        ##########################BBBOX ##########################
        ###########################################################
        bbox_path = os.path.join(rootdir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        self.filename_bbox = {img_file: [] for img_file in self.allimages}
        numImgs = len(self.allimages)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = self.allimages[i]
            self.filename_bbox[key] = bbox
        ###############################################################
        ###############################################################

        self.images = []

        split_identifier = self.split_ids[split]
        for idx, image in enumerate(self.allimages):
            if self.train_test_split[idx] == split_identifier:
                self.images.append(image)
         ###################################################for 80 :20 split##########
         ##########################################################################
        self.split_dir = os.path.join(os.path.normpath(rootdir + os.sep + os.pardir), split)
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        self.images = []
        if len(self.allimages) > len(filenames):
            for image in  filenames:
                self.images.append(image + '.jpg') 
        
        #####################################################################
        #########################################################################
        
        #Creating wrong images set for new stackgan
        self.wrong_images = []
        for idx, image in enumerate(self.images):
            wrong_idx = random.randint(0, len(self.images) - 1)
            while idx == wrong_idx:
                wrong_idx = random.randint(0, len(self.images) - 1)
            self.wrong_images.append(self.images[wrong_idx])

        # read captions to determine max sentence length
        caption_dir = os.path.join(rootdir, "text_c10")
        captionpaths = []
        for subdir in os.listdir(caption_dir):
            for file in os.listdir(os.path.join(caption_dir, subdir)):
                captionpaths.append(os.path.join(caption_dir, subdir, file))

        # Note:
        # see Note in ShapesDataset
        captions = []
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', lines[0])
                    captions.append(line.strip()) # select always the first description as caption for now
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in captions])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos

        # build vocabulary
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.images)
    
    def crop_image(self, img, box):
        #######################cropping important part################
        width, height = img.size
        r = int(np.maximum(box[2], box[3]) * 0.75)
        center_x = int((2 * box[0] + box[2]) / 2)
        center_y = int((2 * box[1] + box[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        ###################################################################
        #imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    
        imsize = self.size * (2 ** (self.num_dis-1))
        image_transform = tt.Compose([
        tt.Resize(int(imsize * 76 / 76)),
        #tt.Resize(int(imsize * 76 / 64)),
        tt.RandomCrop(imsize),
        tt.RandomHorizontalFlip()])
        img = image_transform(img)
        ############################################################
        return img
        ######################################################################
    
    def process_image(self, path, box):
        img_padded_li =[]
        img_res_li = []
        resnet_imgs =[]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB') #output is PIL
            img = self.crop_image(img, box)
            for i in range(cfg.TREE.BRANCH_NUM):
                size = self.size*(2**i)
                img_res = tt.Resize((size,size))(img) #Resize into 64*64
                img_res = tt.ToTensor()(img_res)
                img_res_li.append(img_res)
                img_padded = self.transform(img_res)
                img_padded_li.append(img_padded)
                resnet_i = tt.Resize((224,224))(img) #for resnet
                resnet_imgs.append(self.norm2(resnet_i))
        return img_res_li[0], resnet_imgs, img_padded_li

    def __getitem__(self, idx):
        img_padded_list =[] #list containing image tensors of differnt sizes(64,128,256)
        wg_img_padded_list =[] #list containing wrong image tensor of differnt sizes(64,128,256)
        resnet_imgs_list = []
        img_path = os.path.join(self.image_rootdir, self.images[idx])
        wrong_img_path = os.path.join(self.image_rootdir, self.wrong_images[idx])
        caption_path = self.__change_fileending(img_path)
        #################################################################
        img_key = self.images[idx]
        wg_img_key = self.wrong_images[idx]
        bbox = self.filename_bbox[img_key]
        wg_bbox =self.filename_bbox[wg_img_key]
        img_res, resnet_imgs_list, img_padded_list = self.process_image(img_path, bbox)
        _, _, wg_img_padded_list = self.process_image(wrong_img_path, wg_bbox)
        

        caption = ""
        with open(caption_path) as f:
            captions = f.readlines()
        if len(captions) > 0:
            caption = captions[0].strip() # select always the first description as caption for now

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return resnet_imgs_list, img_padded_list, wg_img_padded_list, img_res, caption_encoded, caption_length

    def __change_fileending(self, path):
        """
        Changes fileending from image to text format
        :return: path with .txt ending instead of image format ending
        """
        if self.image_format in path:
            return path.replace(self.image_format, ".txt").replace("images", "text_c10")
        else:
            raise ValueError("Unexpected image format")

class BirdsDataset2(Image_Caption_Dataset):
    #This is a copy from BirdsDataset1 with class informatin##############
    def __init__(self, rootdir, transform, vocab_builder=BirdsVocabBuilder(), split='train', img_format=".jpg", num_dis =3, img_size =64):
        super(BirdsDataset2, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")
        self.num_dis = num_dis
        self.size = img_size
        self.norm2 = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.train_test_split = []
        with open(os.path.join(rootdir, "train_test_split.txt")) as f:
            lines = f.readlines()
            self.train_test_split = [line.split()[1] for line in lines]

        self.allimages = []
        with open(os.path.join(rootdir, "images.txt")) as f:
            lines = f.readlines()
            self.allimages = [line.split()[1] for line in lines]

        # Sanity check
        assert len(self.allimages) == len(self.train_test_split)
        
        ##########################BBBOX ##########################
        ###########################################################
        bbox_path = os.path.join(rootdir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        self.filename_bbox = {img_file: [] for img_file in self.allimages}
        numImgs = len(self.allimages)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = self.allimages[i]
            self.filename_bbox[key] = bbox
        ###############################################################
        ###############################################################

        self.images = []

        split_identifier = self.split_ids[split]
        for idx, image in enumerate(self.allimages):
            if self.train_test_split[idx] == split_identifier:
                self.images.append(image)
         ###################################################for 80 :20 split##########
         ##########################################################################
        self.split_dir = os.path.join(os.path.normpath(rootdir + os.sep + os.pardir), split)
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        self.images = []
        if len(self.allimages) > len(filenames):
            for image in  filenames:
                self.images.append(image + '.jpg') 
        
        #####################################################################
        #########################################################################
        
        #Creating wrong images set for new stackgan
        self.wrong_images = []
        for idx, image in enumerate(self.images):
            wrong_idx = random.randint(0, len(self.images) - 1)
            while idx == wrong_idx:
                wrong_idx = random.randint(0, len(self.images) - 1)
            self.wrong_images.append(self.images[wrong_idx])

        # read captions to determine max sentence length
        caption_dir = os.path.join(rootdir, "text_c10")
        captionpaths = []
        for subdir in os.listdir(caption_dir):
            for file in os.listdir(os.path.join(caption_dir, subdir)):
                captionpaths.append(os.path.join(caption_dir, subdir, file))

        # Note:
        # see Note in ShapesDataset
        captions = []
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', lines[0])
                    captions.append(line.strip()) # select always the first description as caption for now
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in captions])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos

        # build vocabulary
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.images)
    
    def crop_image(self, img, box):
        #######################cropping important part################
        width, height = img.size
        r = int(np.maximum(box[2], box[3]) * 0.75)
        center_x = int((2 * box[0] + box[2]) / 2)
        center_y = int((2 * box[1] + box[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        ###################################################################
        #imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    
        imsize = self.size * (2 ** (self.num_dis-1))
        image_transform = tt.Compose([
        tt.Resize(int(imsize * 76 / 76)),
        #tt.Resize(int(imsize * 76 / 64)),
        tt.RandomCrop(imsize),
        tt.RandomHorizontalFlip()])
        img = image_transform(img)
        ############################################################
        return img
        ######################################################################
    
    def process_image(self, path, box):
        img_padded_li =[]
        img_res_li = []
        resnet_imgs =[]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB') #output is PIL
            img = self.crop_image(img, box)
            for i in range(cfg.TREE.BRANCH_NUM):
                size = self.size*(2**i)
                img_res = tt.Resize((size,size))(img) #Resize into 64*64
                img_res = tt.ToTensor()(img_res)
                img_res_li.append(img_res)
                img_padded = self.transform(img_res)
                img_padded_li.append(img_padded)
                resnet_i = tt.Resize((224,224))(img) #for resnet
                resnet_imgs.append(self.norm2(resnet_i))
        return img_res_li[0], resnet_imgs, img_padded_li

    def __getitem__(self, idx):
        img_padded_list =[] #list containing image tensors of differnt sizes(64,128,256)
        wg_img_padded_list =[] #list containing wrong image tensor of differnt sizes(64,128,256)
        resnet_imgs_list = []
        class_label=int(self.images[idx][:3])
        img_path = os.path.join(self.image_rootdir, self.images[idx])
        wrong_img_path = os.path.join(self.image_rootdir, self.wrong_images[idx])
        caption_path = self.__change_fileending(img_path)
        #################################################################
        img_key = self.images[idx]
        wg_img_key = self.wrong_images[idx]
        bbox = self.filename_bbox[img_key]
        wg_bbox =self.filename_bbox[wg_img_key]
        img_res, resnet_imgs_list, img_padded_list = self.process_image(img_path, bbox)
        _, _, wg_img_padded_list = self.process_image(wrong_img_path, wg_bbox)
        

        caption = ""
        with open(caption_path) as f:
            captions = f.readlines()
        if len(captions) > 0:
            caption = captions[0].strip() # select always the first description as caption for now

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return resnet_imgs_list, img_padded_list, wg_img_padded_list, img_res, caption_encoded, caption_length, class_label

    def __change_fileending(self, path):
        """
        Changes fileending from image to text format
        :return: path with .txt ending instead of image format ending
        """
        if self.image_format in path:
            return path.replace(self.image_format, ".txt").replace("images", "text_c10")
        else:
            raise ValueError("Unexpected image format")

           
class FlowersDataset1(Image_Caption_Dataset):
    def __init__(self, rootdir, transform, vocab_builder=FlowersVocabBuilder(), split='train', img_format=".jpg", num_dis =3, img_size =64):
        super(FlowersDataset1, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")
        self.num_dis = num_dis
        self.size = img_size
        self.norm2 = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

       
        
        split_classes_train_file_name = os.path.join(rootdir, 'trainvalclasses.txt')
        
        split_classes_test_file_name = os.path.join(rootdir, 'testclasses.txt')
        
        
        self.split_train_classes = []
            
        with open(split_classes_train_file_name) as f:
            lines = f.readlines()
            self.split_train_classes = [line.strip() for line in lines]
            
            
        self.split_test_classes = []
            
        with open(split_classes_test_file_name) as f:
            lines = f.readlines()
            self.split_test_classes = [line.strip() for line in lines]
        
        
        
            

        self.allimages = glob.glob(os.path.join(self.image_rootdir, "*"+img_format))
        
        train_caption_path=[]
        train_class_dict = dict()
        for class1 in self.split_train_classes:
            cpaths = glob.glob(os.path.join(rootdir, 'text_c10', class1,'*.txt'))
            for path in cpaths:
                train_class_dict[ntpath.basename(path)[:-4]] = class1
            train_caption_path += cpaths
        test_caption_path=[]
        test_class_dict = dict()
        for class1 in self.split_test_classes:
            cpaths= glob.glob(os.path.join(rootdir, 'text_c10', class1,'*.txt'))
            for path in cpaths:
                test_class_dict[ntpath.basename(path)[:-4]]= class1
            test_caption_path += cpaths
        

        # Sanity check
        assert len(self.allimages) == len(train_caption_path) + len(test_caption_path)
        
        
        
        

        
        
        if split == 'train':
            self.caption_paths = train_caption_path
            class_dict= train_class_dict
        else:
            self.caption_paths = test_caption_path
            class_dict= test_class_dict
        self.images = []
        self.class_id = list()
        for cpath in self.caption_paths:
            ipath= os.path.join(self.image_rootdir, ntpath.basename(cpath)[:-4] +img_format)
            if ipath in self.allimages:
                self.images.append(ipath)
                k = class_dict[ntpath.basename(ipath)[:-4]]
                self.class_id.append(int(k[6:]))
            

        
         ###################################################for 80 :20 split##########
         
        #########################################################################
        
        #Creating wrong images set for new stackgan
        self.wrong_images = []
        for idx, image in enumerate(self.images):
            wrong_idx = random.randint(0, len(self.images) - 1)
            while self.class_id[idx] == self.class_id[wrong_idx]:
                wrong_idx = random.randint(0, len(self.images) - 1)
            self.wrong_images.append(self.images[wrong_idx])

        # read captions to determine max sentence length
        captionpaths = train_caption_path + test_caption_path
        

        # Note:
        # see Note in ShapesDataset
        captions = []
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', lines[0])
                    captions.append(line.strip()) # select always the first description as caption for now
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in captions])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos

        # build vocabulary
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.images)
    
    def crop_image(self, img):
        #######################cropping important part################
        width, height = img.size
        
        ###################################################################
        #imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    
        imsize = self.size * (2 ** (self.num_dis-1))
        image_transform = tt.Compose([
        tt.Resize(int(imsize * 76 / 76)),
        #tt.Resize(int(imsize * 76 / 64)),
        tt.RandomCrop(imsize),
        tt.RandomHorizontalFlip()])
        img = image_transform(img)
        ############################################################
        return img
        ######################################################################
    
    def process_image(self, path):
        img_padded_li =[]
        img_res_li = []
        resnet_imgs =[]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB') #output is PIL
            img = self.crop_image(img)
            for i in range(cfg.TREE.BRANCH_NUM):
                size = self.size*(2**i)
                img_res = tt.Resize((size,size))(img) #Resize into 64*64
                img_res = tt.ToTensor()(img_res)
                img_res_li.append(img_res)
                img_padded = self.transform(img_res)
                img_padded_li.append(img_padded)
                resnet_i = tt.Resize((224,224))(img) #for resnet
                resnet_imgs.append(self.norm2(resnet_i))
        return img_res_li[0], resnet_imgs, img_padded_li

    def __getitem__(self, idx):
        img_padded_list =[] #list containing image tensors of differnt sizes(64,128,256)
        wg_img_padded_list =[] #list containing wrong image tensor of differnt sizes(64,128,256)
        resnet_imgs_list = []
        img_path =  self.images[idx]
        wrong_img_path = self.wrong_images[idx]
        caption_path = self.caption_paths[idx]
        #################################################################
        
        
        img_res, resnet_imgs_list, img_padded_list = self.process_image(img_path)
        _, _, wg_img_padded_list = self.process_image(wrong_img_path)
        

        caption = ""
        with open(caption_path) as f:
            captions = f.readlines()
        if len(captions) > 0:
            caption = captions[0].strip() # select always the first description as caption for now

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return resnet_imgs_list, img_padded_list, wg_img_padded_list, img_res, caption_encoded, caption_length
    
class FlowerstextDataset(Image_Caption_Dataset):
    #Each image has multiple captions per image, It returns all the captions for birds dataset
    #Created to increase train datasize
    def __init__(self, rootdir, transform, vocab_builder=FlowersVocabBuilder(), split='train', img_format=".jpg"):
        super(FlowerstextDataset, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")
        
        split_classes_train_file_name = os.path.join(rootdir, 'trainvalclasses.txt')
        
        split_classes_test_file_name = os.path.join(rootdir, 'testclasses.txt')
        
        
        self.split_train_classes = []
            
        with open(split_classes_train_file_name) as f:
            lines = f.readlines()
            self.split_train_classes = [line.strip() for line in lines]
            
            
        self.split_test_classes = []
            
        with open(split_classes_test_file_name) as f:
            lines = f.readlines()
            self.split_test_classes = [line.strip() for line in lines]
        
        
        
            

        self.allimages = glob.glob(os.path.join(self.image_rootdir, "*"+img_format))
        
        train_caption_path=[]
        for class1 in self.split_train_classes:
            train_caption_path += glob.glob(os.path.join(rootdir, 'text_c10', class1,'*.txt'))
        test_caption_path=[]
        for class1 in self.split_test_classes:
            test_caption_path += glob.glob(os.path.join(rootdir, 'text_c10', class1,'*.txt'))
        

        # Sanity check
        assert len(self.allimages) == len(train_caption_path) + len(test_caption_path)
        
        
        
        

        
        
        if split == 'train':
            self.caption_paths = train_caption_path
        else:
            self.caption_paths = test_caption_path
        
            

        
        #####################################################################
        ###################################################################

        # read captions to determine max sentence length
        
        captionpaths = train_caption_path + test_caption_path
        

        # Note:
        # see Note in ShapesDataset
        self.captions = []
        for caption_path in self.caption_paths:
            f = open(caption_path , 'r', encoding='utf-8')
            for line in f:
                line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                self.captions.append(line.strip().lower())
            f.close()
        
        allcaptions =[]
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    for line in lines:
                        line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                        allcaptions.append(line.strip()) # select always the first description as caption for now
                        
                        
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in allcaptions])
        
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos
        
        
        #print('the maximum sentence length',self.max_sent_length)
        #print('number of images:', len(self.images), 'number of captions:',len(self.captions))
        # build vocabulary
        print('number of captions in birds dataset', len(self.captions))
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.captions)
    
   
    def __getitem__(self, idx):
        caption = self.captions[idx]
        _ =''

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return _, _, caption_encoded, caption_length

class FlowersDataset2(Image_Caption_Dataset):
    #copied from FlowersDataset1 with class lable
    def __init__(self, rootdir, transform, vocab_builder=FlowersVocabBuilder(), split='train', img_format=".jpg", num_dis =3, img_size =64):
        super(FlowersDataset2, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transform = transform
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, "images")
        self.num_dis = num_dis
        self.size = img_size
        self.norm2 = tt.Compose([
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

       
        
        split_classes_train_file_name = os.path.join(rootdir, 'trainvalclasses.txt')
        
        split_classes_test_file_name = os.path.join(rootdir, 'testclasses.txt')
        
        
        self.split_train_classes = []
            
        with open(split_classes_train_file_name) as f:
            lines = f.readlines()
            self.split_train_classes = [line.strip() for line in lines]
            
            
        self.split_test_classes = []
            
        with open(split_classes_test_file_name) as f:
            lines = f.readlines()
            self.split_test_classes = [line.strip() for line in lines]
        
        
        
            

        self.allimages = glob.glob(os.path.join(self.image_rootdir, "*"+img_format))
        
        train_caption_path=[]
        train_class_dict = dict()
        for class1 in self.split_train_classes:
            cpaths = glob.glob(os.path.join(rootdir, 'text_c10', class1,'*.txt'))
            for path in cpaths:
                train_class_dict[ntpath.basename(path)[:-4]] = class1
            train_caption_path += cpaths
        test_caption_path=[]
        test_class_dict = dict()
        for class1 in self.split_test_classes:
            cpaths= glob.glob(os.path.join(rootdir, 'text_c10', class1,'*.txt'))
            for path in cpaths:
                test_class_dict[ntpath.basename(path)[:-4]]= class1
            test_caption_path += cpaths
        

        # Sanity check
        assert len(self.allimages) == len(train_caption_path) + len(test_caption_path)
        
        
        
        

        
        
        if split == 'train':
            self.caption_paths = train_caption_path
            class_dict= train_class_dict
        else:
            self.caption_paths = test_caption_path
            class_dict= test_class_dict
        self.images = []
        self.class_id = list()
        for cpath in self.caption_paths:
            ipath= os.path.join(self.image_rootdir, ntpath.basename(cpath)[:-4] +img_format)
            if ipath in self.allimages:
                self.images.append(ipath)
                k = class_dict[ntpath.basename(ipath)[:-4]]
                self.class_id.append(int(k[6:]))
            

        
         ###################################################for 80 :20 split##########
         
        #########################################################################
        
        #Creating wrong images set for new stackgan
        self.wrong_images = []
        for idx, image in enumerate(self.images):
            wrong_idx = random.randint(0, len(self.images) - 1)
            while self.class_id[idx] == self.class_id[wrong_idx]:
                wrong_idx = random.randint(0, len(self.images) - 1)
            self.wrong_images.append(self.images[wrong_idx])

        # read captions to determine max sentence length
        captionpaths = train_caption_path + test_caption_path
        

        # Note:
        # see Note in ShapesDataset
        captions = []
        for captionpath in captionpaths:
            with open(captionpath) as f:
                lines = f.readlines()
                if len(lines) > 0:
                    line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', lines[0])
                    captions.append(line.strip()) # select always the first description as caption for now
        max_sent_length = max([len(self.vocab_builder.tokenizer.tokenize(sentence)) for sentence in captions])
        self.max_sent_length = max_sent_length + 2  # + 2 for sos and eos

        # build vocabulary
        self.build_vocab(captionpaths)

    def __len__(self):
        return len(self.images)
    
    def crop_image(self, img):
        #######################cropping important part################
        width, height = img.size
        
        ###################################################################
        #imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    
        imsize = self.size * (2 ** (self.num_dis-1))
        image_transform = tt.Compose([
        tt.Resize(int(imsize * 76 / 76)),
        #tt.Resize(int(imsize * 76 / 64)),
        tt.RandomCrop(imsize),
        tt.RandomHorizontalFlip()])
        img = image_transform(img)
        ############################################################
        return img
        ######################################################################
    
    def process_image(self, path):
        img_padded_li =[]
        img_res_li = []
        resnet_imgs =[]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB') #output is PIL
            img = self.crop_image(img)
            for i in range(cfg.TREE.BRANCH_NUM):
                size = self.size*(2**i)
                img_res = tt.Resize((size,size))(img) #Resize into 64*64
                img_res = tt.ToTensor()(img_res)
                img_res_li.append(img_res)
                img_padded = self.transform(img_res)
                img_padded_li.append(img_padded)
                resnet_i = tt.Resize((224,224))(img) #for resnet
                resnet_imgs.append(self.norm2(resnet_i))
        return img_res_li[0], resnet_imgs, img_padded_li

    def __getitem__(self, idx):
        img_padded_list =[] #list containing image tensors of differnt sizes(64,128,256)
        wg_img_padded_list =[] #list containing wrong image tensor of differnt sizes(64,128,256)
        resnet_imgs_list = []
        img_path =  self.images[idx]
        wrong_img_path = self.wrong_images[idx]
        caption_path = self.caption_paths[idx]
        class_label = self.class_id[idx]
        #################################################################
        
        
        img_res, resnet_imgs_list, img_padded_list = self.process_image(img_path)
        _, _, wg_img_padded_list = self.process_image(wrong_img_path)
        

        caption = ""
        with open(caption_path) as f:
            captions = f.readlines()
        if len(captions) > 0:
            caption = captions[0].strip() # select always the first description as caption for now

        caption_encoded, caption_length = self.vocab_builder.encode_sentences([caption], self.max_sent_length)
        caption_encoded = caption_encoded.squeeze()
        caption_length = caption_length.squeeze()
        return resnet_imgs_list, img_padded_list, wg_img_padded_list, img_res, caption_encoded, caption_length, class_label
    

    

    
    

    
