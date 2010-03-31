'''
Created on Mar 31, 2010
@author: Hua Binh Son
'''

import numpy
import pylab
import sift

class Feature(object):
    def __init__(self, imageId, descriptor, location):
        self.imageId    = imageId;
        self.descriptor = descriptor;
        self.location   = location;

def featureDistance(f1, f2):
    return 1.0 - abs(numpy.dot(f1.descriptor, f2.descriptor));
        
class Image(object):
    '''
    Image class encapsulates image pixel data, SIFT features, and 
    '''
    def __init__(self):
        self.id     = -1;
        self.pixels = None;
        
    def show(self):
        pylab.figure();
        pylab.gray();
        pylab.imshow(self.pixels); #origin='lower'
        pylab.axis('image');

class ImageManager(object):
    # Borg's singleton pattern
    __shared_state  = {}
    
    def __init__(self):
        self.__dict__ = self.__shared_state
        # and whatever else you want in your class -- that's all!
        
        # dictionary to map ID to Image object
        self.totalImages    = 0;
        self.dictIdImage    = {};
        self.dictFileImage  = {};
        
    # Load image from a file and return Image object
    def loadImage(self, file):
        if file in self.dictFileImage:
            return self.dictFileImage[file];
        else:
            # SIFT features
            partName, partDot, partExt = file.rpartition('.');
            keyFile = ''.join(partName + partDot + ("key")); # join tuples to string
            sift.process_image(file, keyFile);
            des, loc  = sift.read_features_from_file(keyFile);
            
            im = Image();
            im.id     = self.totalImages;
            im.pixels = pylab.flipud(pylab.imread(file));            
            im.features = [Feature(im.id, des[i], loc[i]) for i in range(len(des))];
            print "Total features: ", len(im.features)
            
            # add to dictionary
            self.dictFileImage[file]    = im;
            self.dictIdImage[im.id]     = im;
            # increase total images
            self.totalImages += 1;
            print "Total images: ", self.totalImages;
            return im;
    
    def getImageByID(self, id):
        if id in self.dictIdImage:
            return self.dictIdImage[id];
        else:
            return None;
        
class ImageMosaick(object):
    def __init__(self):
        self.images = None
    
    def mosaick(self, imageFiles):
        manager = ImageManager();
        self.images = [manager.loadImage(file) for file in imageFiles];
        
        # insert all features into Kd-tree
        # for each feature find K nearest neighbor match
        # recover H for each pair of images
        # remove outliers using RANSAC
        # 
        
        
    def show(self):
        [im.show() for im in self.images];
        pylab.show();
        
def main():
    imo = ImageMosaick();
    folder  = "./images";
    images  = ["scene.pgm", "box.pgm"];
    imo.mosaick([folder + "/" + image for image in images]);
    imo.show();
    
if __name__ == "__main__":
    main();
        