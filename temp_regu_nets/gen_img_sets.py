"""Generate Tensors for Keras model learning
Example:
python generateImageSets --dataset=data/
"""

from PIL import Image
import numpy as np
import argparse
import glob
import os
import string
import re

train_size = 1000
test_size = 10


def parse_args():
    """
    Handles input of folder with all gifs with different resolution images
    """
    parser = argparse.ArgumentParser(description="dataset folder")
    parser.add_argument('--dataset', dest="dataset", help="dataset path")
    return parser.parse_args()

def gen_GT_HR_sets(path='../data/', tag='face'):
    """
    Generates sets of ground-truth and high-res images for each frame of a gif
    Specifically each set is a 3-element list of 3-D matrices representing the
        GT and HR images.
    Each set is structured as [GT, HR, GT2]] with GT being at frame n,
        and HR at frame n+1, and GT2 at frame n+1

    Input:
    path :  file path to directory with folders of gifs each with subfolders for
            lr/hr/gt images in numerical order of frames. 00000.jpg is the first
            frame
    Returns:
    5-D Tensor with a list of sets of 3D matrices
    """

    hrGIFs = glob.glob(path + 'hr_imgs/' + tag + "/*/")
    hrGIFs.sort(key=lambda f: int(filter(str.isdigit, f)))
    hrGIFs_train = hrGIFs[:train_size]
    hrGIFs_test = hrGIFs[train_size:train_size+test_size]
    lrGIFs = glob.glob(path + 'lr_imgs/' + tag + "/*/")
    lrGIFs.sort(key=lambda f: int(filter(str.isdigit, f)))
    lrGIFs_train = lrGIFs[:train_size]
    lrGIFs_test = lrGIFs[train_size:train_size+test_size]
    srGIFs = glob.glob(path + 'bi_imgs/' + tag + "/*/")
    srGIFs.sort(key=lambda f: int(filter(str.isdigit, f)))
    srGIFs_train = srGIFs[:train_size]
    srGIFs_test = srGIFs[train_size:train_size+test_size]

    if not (len(hrGIFs) == len(lrGIFs) and len(lrGIFs) == len(srGIFs)):
        raise ValueError('GIF Imcomplete!')
    else:
        print 'Folder check complete!'

    gif_id = []
    frame_num = []

    print 'Generating Training Set ...'

    train_dataset = []
    for idx, gifFolder in enumerate(hrGIFs_train):

        # print idx, gifFolder
        hrImages = os.listdir(path + 'hr_imgs/' + tag + "/" + str(idx))
        srImages = os.listdir(path + 'bi_imgs/' + tag + "/" + str(idx))
        frame_num.append(len(hrImages))

        #Take up to 2nd to last image
        for pos in xrange(len(hrImages)-2):

            gif_id.append(idx)

            hrImage1 = hrImages[pos]
            srImage = srImages[pos+1]
            hrImage2 = hrImages[pos+2]
            hrImage_y = hrImages[pos+1]

            imageDir = [hrImage1, srImage, hrImage2, hrImage_y]
            # print imageDir
            imageSet = imageDir
            folderSet = ["hr_imgs", "bi_imgs", "hr_imgs", "hr_imgs"]

            # gtImage  = Image.open(path+"/"+gifFolder+"/"+"gt"+"/"+gtImage)
            # hrImage  = Image.open(path+"/"+gifFolder+"/"+"gt"+"/"+hrImage)
            # gtImage2 = Image.open(path+"/"+gifFolder+"/"+"gt"+"/"+gtImage2)

            for i, img in enumerate(imageDir):
                # print path + folderSet[i] + '/' + tag + '/' + str(idx) + '/' + imageDir[i]
                imageSet[i]=Image.open(path + folderSet[i] + '/' + tag + '/' + str(idx) + '/' + imageDir[i])
                imageSet[i]=imageSet[i].convert('RGB')
                imageSet[i]=np.asarray(imageSet[i], dtype=np.float32)

            # #If one image not RGB, assume all aren't
            # if gtImage.mode != 'RGB':
            #     gtImage  = gtImage.convert('RGB')
            #     hrImage  = hrImage.convert('RGB')
            #     gtImage2 = gtImage2.convert('RGB')
            #
            # gtImage  = np.asarray(gtImage, dtype=np.float32)
            # hrImage  = np.asarray(hrImage, dtype=np.float32)
            # gtImage2 = np.asarray(gtImage2, dtype=np.float32)

            train_dataset.append(imageSet)

    print 'Generating Test Set ...'

    test_dataset = []
    for idx, gifFolder in enumerate(srGIFs_test):

        # print idx, gifFolder
        hrImages = os.listdir(path + 'hr_imgs/' + tag + "/" + str(idx + train_size))
        srImages = os.listdir(path + 'bi_imgs/' + tag + "/" + str(idx + train_size))
        frame_num.append(len(srImages))

        #Take up to 2nd to last image
        for pos in xrange(len(srImages)-2):

            gif_id.append(idx + train_size)

            srImage1 = srImages[pos]
            srImage = srImages[pos+1]
            srImage2 = srImages[pos+2]
            hrImage_y = hrImages[pos+1]

            imageDir = [srImage1, srImage, srImage2, hrImage_y]
            # print imageDir
            imageSet = imageDir
            folderSet = ["bi_imgs", "bi_imgs", "bi_imgs", "hr_imgs"]

            for i, img in enumerate(imageDir):
                # print path + folderSet[i] + '/' + tag + '/' + str(idx) + '/' + imageDir[i]
                imageSet[i]=Image.open(path + folderSet[i] + '/' + tag + '/' + str(idx + train_size) + '/' + imageDir[i])
                imageSet[i]=imageSet[i].convert('RGB')
                imageSet[i]=np.asarray(imageSet[i], dtype=np.float32)

            test_dataset.append(imageSet)

    # print gif_id
    # print len(frame_num)

    return np.array(train_dataset), np.array(test_dataset), gif_id, frame_num

def generate_GT_HR_attention_sets(path, steps=1):
        """
        Generates sets of ground-truth and high-res images for each frame of a gif
        Specifically each set is a steps+2-element list of 3-D matrices representing the
            GT and HR images.
        Each set is structured as [[GT], HR, GT2]] with GT being at frame n:n+steps-1,
            and HR at frame n+steps, and GT2 at frame n+steps

        Input:
        path :  file path to directory with folders of gifs each with subfolders for
                lr/hr/gt images in numerical order of frames. 00000.jpg is the first
                frame
        steps : Number of images that are grouped together
        Returns:
        5-D Tensor with a list of sets of 3D matrices
        """
        dataset = []
        if steps<1:
            raise ValueError('Input var steps must be greater than 0')
        for gifFolder in os.listdir(path):

            subdirs = os.listdir(path+"/"+gifFolder)
            if not('gen' in subdirs and 'lr' in subdirs and 'hr' in subdirs):
                raise ValueError('could not find gt lr and hr subdirs in %s'
                                    % (path+"/"+gifFolder))
            hrImages = os.listdir(path+"/"+gifFolder+"/"+"hr")
            lrImages = os.listdir(path+"/"+gifFolder+"/"+"lr")
            gtImages = os.listdir(path+"/"+gifFolder+"/"+"gen")

            #Take up to 2nd to last image
            #print("gifFolder: %s, files: %d" % (gifFolder, len(gtImages)-steps))
            for pos in xrange(len(gtImages)-steps):
                gtImage  = gtImages[pos:pos+steps]
                hrImage  = hrImages[pos+steps]
                gtImage2 = gtImages[pos:pos+steps+1]


                imageSet = gtImage
                imageSet.append(hrImage)
                for gtI in gtImage2:
                    imageSet.append(gtI)
                #imageSet = [gtImage, hrImage, gtImage2]
                folderSet= ["gen","hr","gen"]


                # gtImage  = Image.open(path+"/"+gifFolder+"/"+"gt"+"/"+gtImage)
                # hrImage  = Image.open(path+"/"+gifFolder+"/"+"gt"+"/"+hrImage)
                # gtImage2 = Image.open(path+"/"+gifFolder+"/"+"gt"+"/"+gtImage2)

                for i,img in enumerate(imageSet):

                    if i==((len(imageSet)-1.)/2):
                        folder="hr"
                    else:
                        folder="gen"

                    imageSet[i]=Image.open(path+"/"+gifFolder+"/"+folder+"/"+imageSet[i])
                    imageSet[i]=imageSet[i].convert('RGB')
                    imageSet[i]=np.asarray(imageSet[i], dtype=np.float32)

                # #If one image not RGB, assume all aren't
                # if gtImage.mode != 'RGB':
                #     gtImage  = gtImage.convert('RGB')
                #     hrImage  = hrImage.convert('RGB')
                #     gtImage2 = gtImage2.convert('RGB')
                #
                # gtImage  = np.asarray(gtImage, dtype=np.float32)
                # hrImage  = np.asarray(hrImage, dtype=np.float32)
                # gtImage2 = np.asarray(gtImage2, dtype=np.float32)
                dataset.append(imageSet)
        return np.array(dataset)

if __name__=='__main__':

    args = parse_args()
    dataFolder = args.dataset
    generate_GT_HR_sets(dataFolder)

    