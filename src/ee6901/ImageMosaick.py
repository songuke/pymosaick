'''
Created on Mar 31, 2010
@author: Hua Binh Son
'''

import numpy as np
import pylab
import sift
import KdtreeCustom
import time

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
    def __init__(self, id, pixels, locations, descriptors):
        self.id     = id;
        self.pixels = pixels;
        self.descriptors = descriptors;
        self.locations = locations;
        
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
            id     = self.totalImages;
            pixels = pylab.flipud(pylab.imread(file));            
            
            # SIFT features
            partName, partDot, partExt = file.rpartition('.');
            keyFile = ''.join(partName + partDot + ("key")); # join tuples to string
            sift.process_image(file, keyFile);
            loc, des  = sift.read_features_from_file(keyFile);
            #im.features = [Feature(im.id, des[i], loc[i]) for i in range(len(des))];
            #print "Total features: ", len(im.features)
            
            im = Image(id, pixels, loc, des);
            
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
        
class ImageMatch(object):
    def __init__(self):
        self.image1 = None;
        self.image2 = None;
        self.locs1 = [];
        self.locs2 = [];
    
        
class ImageMosaick(object):
    def __init__(self):
        self.images = None
    
    def mosaick(self, imageFiles):
        start = time.clock();
        
        manager = ImageManager();
        self.images = [manager.loadImage(file) for file in imageFiles];
        
        elapsed = time.clock() - start;
        print "SIFT: ", elapsed; 
        
        start = time.clock();
        
        self.match = [[ImageMatch()] * len(self.images)] * len(self.images);
        for i in range(len(self.images) - 1):
            for j in range(i + 1, len(self.images)):
                self.match[i][j].image1 = self.images[i];
                self.match[i][j].image2 = self.images[j];
        
        for i in range(len(self.images) - 1):
            # build a kd-tree of n - 1 other images' features
            # feature array
            features = [];
            locations = [];
            imageIds = [];
            for j in range(i + 1, len(self.images)):
                im = self.images[j];
                features.extend(im.descriptors);
                locations.extend(im.locations); # a list of array(4)
                # image index of each features
                imageIds.extend([im.id for m in range(len(im.descriptors))]);
            
            # insert all features into Kd-tree
            kdtree = KdtreeCustom.KDTree(features, leafsize=64);
           
            # for each feature find K nearest neighbor match
            knearest = 2; # minimum is 2. 1 works not well as it's a hard threshold.
            distRatio = 0.8;
            for n in range(len(self.images[i].descriptors)):
                #print n, "."
                ft = self.images[i].descriptors[n];
                #print n, "-"
                dist, idx = kdtree.query(ft, knearest);
                #print dist
                
                # not accept duplicated items
                #idxUnique, indices = np.unique1d(idx, return_index=True);
                if knearest > 1:
                    matchedImages = set();
                    # the difference between the best and second match should not be larger than 0.8
                    for j in range(knearest - 1):
                        if dist[j] < distRatio * dist[j+1]:
                            m = imageIds[idx[j]];
                            if m not in matchedImages:
                                matchedImages.add(m);
                                self.match[i][m].locs1.append(self.images[i].locations[n]);
                                self.match[i][m].locs2.append(locations[idx[j]]);          
                else:
                    if dist < 0.35:
                        m = imageIds[idx];
                        self.match[i][m].locs1.append(self.images[i].locations[n]);
                        self.match[i][m].locs2.append(locations[idx]);  
            #print match[n];
            #print [imageIds[match[n][j]] for j in range(knearest)];
        
        elapsed = time.clock() - start;
        print "Kdtree: ", elapsed; 
        
        # recover H for each pair of images
        # remove outliers using RANSAC
        # 
        
        
    def show(self):
        #[im.show() for im in self.images];
        
        pylab.figure(0);
        scores = range(len(self.match[0][1].locs1));
        #print self.match[0][1].locs1;
        #print self.match[0][1].locs1
        sift.plot_matches_2(self.images[0].pixels, self.images[1].pixels, self.match[0][1].locs1, self.match[0][1].locs2, scores);
        pylab.show();
        
def main():
    imo = ImageMosaick();
    folder  = "./images";
    images  = ["scene.pgm", "basmati.pgm"];
    imo.mosaick([folder + "/" + image for image in images]);
    imo.show();
    
if __name__ == "__main__":
    main();
        