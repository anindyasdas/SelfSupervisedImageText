import os
import matplotlib.pyplot as plt
import csv
from models.utils1 import compute_bleu
from textwrap import wrap
from torchvision import transforms as tt

TEXT_TEXT_CSV_HEADER = ["idx", "generated", "original", "bleu_score"]

class ResultWriter():

    def __init__(self, outputdir):
        self.outputdir = outputdir
        self.imageoutputs = os.path.join(outputdir, "images")
        self.textoutputs = os.path.join(outputdir, "captions.csv")
        self.examplecount = 0

        if not os.path.exists(self.outputdir):
                os.mkdir(self.outputdir)

    def write_image_with_text(self, image, text):
        """
        Write and image together with a text. Can be used to store the results of an image to text run
        or vice versa
        :param image: the image to save
        :param text: the text to save
        """
        if not os.path.exists(self.imageoutputs):
            os.mkdir(self.imageoutputs)

        img_name = "img" + str(self.examplecount)

        plt.figure()
        plt.title('\n'.join(wrap(text, 60))) #wrapping the text with length length 60 
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(self.imageoutputs, img_name), bbox_inches='tight') #otherwise it will get truncated
        plt.close()
        self.examplecount += 1

    def write_images(self, gen_img, label_img):
        """
        Write images together in one figure. Can be used to store the results of an image to image run.
        :param gen_img: the generated images
        :param label_img: the input/ label image
        """
        if not os.path.exists(self.imageoutputs):
            os.mkdir(self.imageoutputs)

        img_name = "img" + str(self.examplecount)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(gen_img)
        ax1.set_title('generated')
        ax1.axis('off')
        ax2.imshow(label_img)
        ax2.set_title('original')
        ax2.axis('off')
        plt.savefig(os.path.join(self.imageoutputs, img_name))
        plt.close()
        self.examplecount += 1
    
    def write_images1(self, gen_img):
        """
        Write images together in one figure. Can be used to store the results of an image to image run.
        :param gen_img: the generated images
        :param label_img: the input/ label image
        """
        if not os.path.exists(self.imageoutputs):
            os.mkdir(self.imageoutputs)
        img_name = "img" + str(self.examplecount) +'.jpg'
        img = tt.ToPILImage()(gen_img)
        img = tt.Resize((256,256))(img)#Resize it to 256 *256
        img.save(os.path.join(self.imageoutputs, img_name), format='JPEG')
        self.examplecount += 1
        
    def write_images3(self, gen_img1, gen_img2,gen_img3, label_img):
        """
        Write images together in one figure. Can be used to store the results of an image to image run.
        :param gen_img: the generated images
        :param label_img: the input/ label image
        """
        if not os.path.exists(self.imageoutputs):
            os.mkdir(self.imageoutputs)

        img_name = "img" + str(self.examplecount)
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        ax1.imshow(gen_img1)
        ax1.set_title('generated1')
        ax1.axis('off')
        ax2.imshow(gen_img2)
        ax2.set_title('generated2')
        ax2.axis('off')
        ax3.imshow(gen_img3)
        ax3.set_title('generated3')
        ax3.axis('off')
        ax4.imshow(label_img)
        ax4.set_title('original')
        ax4.axis('off')
        plt.savefig(os.path.join(self.imageoutputs, img_name))
        plt.close()
        self.examplecount += 1

    def write_texts(self, gen_text, label_text):
        """
        Write two texts together in a csv-file. Can be used to store the results of a text to text run.
        :param gen_text: the generated text
        :param label_text: the input/ label text
        """
        if not os.path.exists(self.imageoutputs):
            os.mkdir(self.imageoutputs)

        textout = os.path.join(self.outputdir, "textpairs.txt")
        idx = self.examplecount
        with open(textout, 'w') as f:
            csvwriter = csv.writer(f)
            if os.stat(textout).st_size == 0:
                csvwriter.writerow(TEXT_TEXT_CSV_HEADER)
            bleu_score = compute_bleu(gen_text, label_text)
            csvwriter.writerow([idx, gen_text, label_text, bleu_score])

        self.examplecount += 1
