import os
from torchvision import transforms as tt
from skimage import io
from torch.utils.data import Dataset
from text import TextLoader
from matplotlib.pyplot import imread
from torch.nn.modules.padding import ConstantPad2d
import csv
from PIL import Image
from data.vocab import ShapesVocabBuilder, BirdsVocabBuilder

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
    def __init__(self, rootdir, transforms, vocab_builder=BirdsVocabBuilder(), split='train', img_format=".jpg"):
        super(BirdsDataset, self).__init__(vocab_builder)
        self.split_ids = {
            'train': "0",
            'val': "1"
        }
        self.image_format = img_format
        self.transforms = transforms
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

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_rootdir, self.images[idx])
        caption_path = self.__change_fileending(img_path)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transforms(img)
        img_padded = ConstantPad2d(80, 0)(img)

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
