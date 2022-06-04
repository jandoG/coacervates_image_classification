"""
This file contains all the required functions to run the jupyter notebooks. Keep it next to the notebooks ALWAYS! 

Author: Gayathri Nadar 
Date: 2021-09-24
"""

import numpy as np 
import pandas as pd 
import os
from os.path import join
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import tifffile 
import random
import mahotas
import pickle
import skimage 
from skimage import data, segmentation, feature, future

def getListofPathsFromTxtFile(filepath):
    """
    Reads the txt file and returns paths that end in .tif as a list. 
    
    Checking is already done to see if file exists, and if not, an error is printed out. 
    Only files that exist are added to the list and returned.
    
    params:
    filepath: str path to text file 
    
    returns:
    filepaths: list of paths to tif files     
    """
    with open(filepath) as f:
        filepaths = []
        
        for line in f.readlines():
            l = str(line.strip())                             # remove whitespace, newline
            if os.path.exists(l) and l.endswith(".tif"):      # find tif files 
                filepaths.append(l)
            else:
                print("The following file: {} was not found, check the path in txt file!\n".format(l))    
    f.close()
    print("Done reading paths from text file: ", os.path.basename(filepath))
    
    return filepaths

def readImagesfromPathsList(pathslist, downsize = True):
    """
    Reads images from a list of file paths and returns an image array. 
    Downsizes image if downsize = True (default). Image is downsized by factor 4.
    New image dimensions after downsizing = width/4 x height/4. 
    Dimensions should be divisible by 4 (ideally). 
    
    params:
    pathslist: list of paths to .tif files 
    downsize: bool 
    
    returns:
    images_array: numpy nd array of images. Size = (no files, X, Y) 
    
    """
    images = []
    
    print("Starting to read images in list. This might take time...")
    
    for p in pathslist:
        if os.path.exists(p):
            image = tifffile.imread(p)
            shape = image.shape
            if downsize:
                im = cv2.resize(image, dsize=(int(shape[0]/4), int(shape[1]/4)), interpolation=cv2.INTER_CUBIC)
            else:
                im = image 
            
            images.append(np.asarray(im))
            images_array = np.array(images)
        else:
            print("The following file: {} was not found, check the path and try again!\n")
            
    print("Reading files - done!\n")
        
    return images_array

def readImagesfromPathsListandResize(pathslist, downsize = True, newsize = (512, 512)):
    """
    Reads images from a list of file paths and returns an image array. 
    Resizes image if downsize = True (default). Image is resized to (512, 512) or newsize 
    
    params:
    pathslist: list of paths to .tif files 
    downsize: bool 
    
    returns:
    images_array: numpy nd array of images. Size = (no files, X, Y) 
    
    """
    images = []
    
    print("Starting to read images in list. This might take time...")
    
    for p in pathslist:
        if os.path.exists(p):
            image = tifffile.imread(p)
            shape = image.shape
            if downsize:
                im = cv2.resize(image, dsize=newsize, interpolation=cv2.INTER_CUBIC)
            else:
                im = image 
            
            images.append(np.asarray(im))
            images_array = np.array(images)
        else:
            print("The following file: {} was not found, check the path and try again!\n")
            
    print("Reading files - done!\n")
        
    return images_array


def getRandomIndices(maxvalue, no_samples, seed = 5):
    """
    Samples the range of values till maxvalue and returns no_samples number of unique values. 
    
    params: 
    maxvalue: int 
    no_samples: int 
    
    returns:
    indices: list
    """
    random.seed(seed)
    indices = random.sample(range(maxvalue), no_samples)
    
    return indices

def getFilename(listfiles, keyword):
    """
    Finds list entry which has the word defined by 'keyword' in it. 
    Expects list of strings. Each string is checked whether it has the 'keyword' in it. 
    If true, the strings are added to the list. The first value of the list i.e. at idx 0 is returned. 
    Because, we expect only three txt files in the folder and each has unique keyword.
    
    params: 
    listfiles: list of strings of filenames e.g. ['droplet_paths.txt', 'nondroplet_paths.txt']
    keyword: str identifier of file e.g. 'droplet' or 'nondroplet'    
    """
    filename = [ff for ff in listfiles if keyword in ff]
    if len(filename) != 0:
        fname = filename[0]
        return fname
    else:
        print("File with keyword='{}' not found! Check filename and specify the correct keyword.\n".format(keyword))
        return None 

def getNSamples(list_samples, no_samples = 10, seed = 0):
    """
    Choose and return N samples from a list. 
    The no_samples should strictly be <= no of samples in the list 
    If no of samples to be picked is higher than the available number of samples, an error is thrown.
    
    params: 
    list_samples: list of strings of file paths 
    no_samples: int no of samples to be picked 
    
    returns: 
    samples: list of N values chosen from the list_samples    
    """
    assert no_samples <= len(list_samples), "No of samples chosen is higher than available number of samples, \
                                        reduce the no of samples to be picked!"
    
    indices = getRandomIndices(len(list_samples), no_samples, seed)
    
    samples = [list_samples[i] for i in indices]
    
    return samples 

def createLabelsArray(length, labelname):
    """
    Create a numpy array of labels for the images. 
    Shape of array = (no of samples, 1)   
    
    params: 
    length: int number of labels 
    labelname: str label name for the images e.g. 'droplet', 'aggregates'
    
    returns: 
    labelarray: numpy array of shape (length, 1) 
    each entry in labelarray is `labelname`
    """
    labelarray = np.full(shape=(length), fill_value=labelname)
    
    return labelarray 

def getMultiscaleFeature(image, sigma_min= 1, sigma_max= 16):
    """
    Create a feature vector by computing multiscale basic image features: edge, intensity, edges, texture. 
    Scales defined by sigma_min and sigma_max 
    Details: https://scikit-image.org/docs/dev/api/skimage.feature.html#multiscale-basic-features 
    
    params: 
    image: numpy array 
    sigma_min, sigma_max: int scale range 
    
    returns: 
    feature_vector: numpy array of shape image.shape + n_features   
    """
    feature_vector = feature.multiscale_basic_features(image, intensity=True, edges=False, texture=True,
        sigma_min=sigma_min, sigma_max=sigma_max)
    
    return feature_vector

def getHaralickFeature(image):
    """
    Create a Haralick feature vector. 
    More info https://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.haralick 
    We compute the mean of the axis=0, hence the output is a 1D array. 
    
    params: 
    image: numpy array 
    
    returns: 
    feature_vector: numpy array of shape (13,) 
    """
    fv_haralick = mahotas.features.haralick(image).mean(axis=0)
    return fv_haralick

def getHistogramFeature(image, mask=None):
    """
    Create a histogram feature vector. 
    
    params: 
    image: numpy array 
    
    returns: 
    fv_hist: 1D numpy array  
    """
    bins = 256
    rangemax = 65535
    hist  = cv2.calcHist([image], [0], None, [bins], [0, rangemax])
    
    # normalize the histogram
    cv2.normalize(hist, hist)
    fv_hist = hist.flatten()
    
    return fv_hist

def getMomentsFeature(image):
    """
    Create a moments feature vector
    
    params: 
    image: numpy array 
    
    returns: 
    fv_moments: 1D numpy array  
    """
    fv_moments = cv2.HuMoments(cv2.moments(image)).flatten()
    return fv_moments
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    