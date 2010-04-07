'''
Created on Mar 31, 2010
@author: Hua Binh Son
'''

import os
import numpy as np
from scipy import linalg
import scipy as sp
import pylab
import sift
import KdtreeCustom
import time
import heapq

class Feature(object):
    def __init__(self, imageId, descriptor, location):
        self.imageId    = imageId;
        self.descriptor = descriptor;
        self.location   = location;
        self.H          = None;
        self.mosaickShape = None;

#def featureDistance(f1, f2):
    #return 1.0 - abs(numpy.dot(f1.descriptor, f2.descriptor));
        
class Image(object):
    '''
    Image class encapsulates image pixel data, SIFT features, and 
    '''
    def __init__(self, id, pixels, locations, descriptors):
        self.id     = id;
        self.pixels = pixels;
        self.descriptors = descriptors;
        self.locations = locations;
        self.shape = np.shape(self.pixels);
        
    def interpolate(self, x, y):
        h, w = self.shape;
        
        x0 = np.floor(x);
        x1 = x0 + 1;
        y0 = np.floor(y);
        y1 = y0 + 1;
        
        if x0 < 0:  x0 = 0;
        if x0 >= w: x0 = w - 1;
        if x1 < 0:  x1 = 0;
        if x1 >= w: x1 = w - 1;
        if y0 < 0:  y0 = 0;
        if y0 >= h: y0 = h - 1;
        if y1 < 0:  y1 = 0;
        if y1 >= h: y1 = h - 1;
        
        s = x - x0;
        t = y - y0;
        
        # A --- B
        # | E   |
        # C --- D
        
        colorA = self.pixels[y0, x0];
        colorB = self.pixels[y0, x1];
        colorC = self.pixels[y1, x0];
        colorD = self.pixels[y1, x1];
                
        colorAB = (1 - s) * colorA + s * colorB;
        colorCD = (1 - s) * colorC + s * colorD;
        
        color = (1 - t) * colorAB + t * colorCD;
        return color; 
        
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
            
            if os.path.exists(keyFile) == False:            
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
        
    def __homography__(self, idx):
        """
        Compute homography matrix to transform image 2 to image 1.
        idx: index to a subset of correspondences for homography computation.
        Return a 3x3 matrix [h11, h12, h13; h21, h22, h23; h31, h32, h33].
        """
        # each correspondence gives two equations
        nbEqns = len(idx) * 2;
        A = np.matrix(np.zeros((nbEqns, 9)));
        #b = np.zeros((nbEqns, 1));
        for i in range(len(idx)):
            p1 = self.locs1[idx[i]]; p1x = p1[0]; p1y = p1[1];
            p2 = self.locs2[idx[i]]; p2x = p2[0]; p2y = p2[1];
            A[2 * i, :] = [p2x, p2y, 1, 0, 0, 0, -p1x * p2x, -p1x * p2y, -p1x];
            A[2 * i + 1, :] = [0, 0, 0, p2x, p2y, 1, -p1y * p2x, -p1y * p2y, -p1y];
        # solve Ah = 0 using SVD
        U, s, Vh = linalg.svd(A);
        # h is the last column of V, or last row of Vh.
        h = Vh[8, :];        
        #print h
        #print h        
        #print A * h        
        #print type(A)
        return np.matrix(np.reshape(h, (3, 3), 'C')); # C means C matrix, row-major.        
        
    def ransac(self):
        # take 4 random correspondences
        size = 4;
                
        # RANSAC
        iters = 200;
        maxInliers = 0;
        for i in range(iters):
            idx = np.random.randint(0, len(self.locs1) - 1, size);
            """
            locs1 = [];
            locs2 = [];
            for j in idx:
                locs1.append(self.locs1[j]);
                locs2.append(self.locs2[j]);
            """
            H = self.__homography__(idx);
            #print H;
            
            # check for consistency
            mask = self.__checkConsistency__(H, 0.5);
            newInliers = np.sum(mask);
            if newInliers > maxInliers:
                maxInliers = newInliers;
                bestMask = mask;
                
        # recompute H with all inliers
        locs1 = [];
        locs2 = [];
        for i in range(len(self.locs1)):
            if bestMask[i] == 1:
                locs1.append(self.locs1[i]);
                locs2.append(self.locs2[i]);
        # only store inliers  
        self.locs1 = locs1;
        self.locs2 = locs2;  
        idx = range(len(self.locs1));            
        H = self.__homography__(idx);        
        print "Best homography: ", H
        print "Inliers/Outliers: ", np.sum(bestMask), "/", len(bestMask) - np.sum(bestMask);
        # store homography matrix
        self.H = H; 
                    
    def __checkConsistency__(self, H, eps=1):
        """
        eps: the tolerance error (in pixel) of the homography estimation
        
        Return an array indicating inliers. 
        0 : outlier. 1 : inlier.
        """
        # map from image2 to image1
        eps2 = eps**2; # tolerance 2 pixels
        mask = np.ones(len(self.locs1));
        for i in range(len(self.locs1)):
            #p2 = np.reshape(np.array([self.locs2[i][0], self.locs2[i][1], 1]), (3, 1));            
            p2 = np.matrix([ [self.locs2[i][0]], [self.locs2[i][1]], [1] ]);
            p1 = H * p2;
            p1 = p1 / p1[2]; # convert back to non-homogeneous
            
            # compute the distance of p1 and original locs1
            dist = (p1[0] - self.locs1[i][0])**2 + (p1[1] - self.locs1[i][1])**2;
            if dist > eps2:
                mask[i] = 0; # outlier

            #print type(p2)
            #print type(p1)
            #print p2
            #print p1
            # 
            #return
        return mask
    def show(self):        
        scores = range(len(self.locs1));
        sift.plot_matches_2(self.image1.pixels, self.image2.pixels, self.locs1, self.locs2, scores);
        
class ImageMosaick(object):
    def __init__(self):
        self.images = None
        self.match = None
        self.bundle = []
        self.shape = None
        
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
                # join two list -> use extend
                features.extend(im.descriptors);
                locations.extend(im.locations); # a list of array(4)
                # image index of each features
                imageIds.extend([im.id for m in range(len(im.descriptors))]);
            
            # insert all features into Kd-tree
            kdtree = KdtreeCustom.KDTree(features, leafsize=512);
           
            # for each feature find K nearest neighbor match
            knearest = 2; # minimum is 2. 1 works not well as it's a hard threshold.
            distRatio = 0.8;
            
            """
            start = time.clock();
            dists, idxs = kdtree.query(self.images[i].descriptors, knearest, eps=0.01, distance_upper_bound=0.3);
            elapsed = time.clock() - start;
            print "Query: ", elapsed
            return;
            """
            
            for n in range(len(self.images[i].descriptors)):
                
                ft = self.images[i].descriptors[n];
                
                #start = time.clock();
                dist, idx = kdtree.query(ft, knearest, eps=0.01, distance_upper_bound=0.1); # prune more? 
                #elapsed = time.clock() - start;
                #print "Query: ", elapsed
                #print dist
                
                # not accept duplicated items
                #idxUnique, indices = np.unique1d(idx, return_index=True);
                
                #start = time.clock();
                if knearest > 1:
                    matchedImages = set();
                    # the difference between the best and second match should not be larger than 0.8
                    for j in range(knearest - 1):
                        if dist[j] < distRatio * dist[j+1]:
                            m = imageIds[idx[j]];
                            if m not in matchedImages:
                                matchedImages.add(m);
                                # add each element of a list to a list -> append
                                self.match[i][m].locs1.append(self.images[i].locations[n]);
                                self.match[i][m].locs2.append(locations[idx[j]]);          
                else:
                    if dist < 0.35:
                        m = imageIds[idx];
                        self.match[i][m].locs1.append(self.images[i].locations[n]);
                        self.match[i][m].locs2.append(locations[idx]);
                        
                #elapsed = time.clock() - start;
                #print "Remaining: ", elapsed  
            #print match[n];
            #print [imageIds[match[n][j]] for j in range(knearest)];
        
        elapsed = time.clock() - start;
        print "Kdtree: ", elapsed; 
        
        # recover H for each pair of images
        # and use RANSAC to reject outliers
        start = time.clock();
        for i in range(len(self.images) - 1):
            for j in range(i + 1, len(self.images)): 
                self.match[i][j].ransac();
        elapsed = time.clock() - start;
        print "RANSAC: ", elapsed;
        
        # find global transform to reference images
        ref = 0;
        start = time.clock();
        self.findGlobalTransform(ref); 
        elapsed = time.clock() - start;
        print "Global transform: ", elapsed;
        
        # perform stitching
        start = time.clock();
        self.stitch(ref);
        elapsed = time.clock() - start;
        print "Stitch: ", elapsed; 
        
    def findGlobalTransform(self, ref = 0):
        """
        Find global transformation matrix for each image to the reference image.
        Input: 
            ref: reference image index. Default is 0.
        """
        # reference image is image[0]        
        self.images[ref].H = np.matrix(np.eye(3));
        
        # global mosaick size
        gh, gw = self.images[ref].shape;
        gxmin = 0; gxmax = gw - 1;
        gymin = 0; gymax = gh - 1;
        
        for im in self.images:
            if im.id == ref: continue;
            
            # find a path to imageRef
            path = self.findPath(im.id, ref);
            # accumulate H
            A = np.matrix(np.eye(3));
            
            h = 0; w = 0;
            for j in path:
                if j < im.id:
                    H = self.match[j][im.id].H;
                else:
                    # inverse
                    H = self.match[im.id][j].H.I;
                    
                # find xmin and ymin
                if h == 0:
                    h, w = im.shape;            
                    
                hj, wj = self.images[j].shape;
                # project to image[j]
                corners = np.matrix([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]);
                xmin = 0; xmax = wj - 1;
                ymin = 0; ymax = hj - 1;
                for c in corners:
                    p = np.matrix(c.reshape((3, 1)));
                    q = np.ravel(H * p);
                    q[0] /= q[2];
                    q[1] /= q[2];
                    xmin = np.amin([xmin, q[0]]);
                    ymin = np.amin([ymin, q[1]]);
                    xmax = np.amax([xmax, q[0]]);
                    ymax = np.amax([ymax, q[1]]);
                
                # update the new image size
                w = np.ceil(xmax - xmin);
                h = np.ceil(ymax - ymin);
                
                # translate to new origin (xmin, ymin)
                #T = np.matrix([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]]);
                #A = A * H * T;
                A = H * A;
                
            # store global homography for each image
            im.H = A;
            #im.shape = (h, w);
            
            # find global image size
            gxmin = np.amin([gxmin, xmin]);
            gymin = np.amin([gymin, ymin]);
            gxmax = np.amax([gxmax, xmax]);
            gymax = np.amax([gymax, ymax]);
        
        # return the global size
        gw = np.ceil(gxmax - gxmin);
        gh = np.ceil(gymax - gymin);
        T = np.matrix([[1, 0, -gxmin], [0, 1, -gymin], [0, 0, 1]]);
        for im in self.images:
            im.H = T * im.H; 
        
        self.shape = (gh, gw);
    
    def findPath(self, src, dest):
        # dijkstra shortest path from src to dest
        visited = [False]   * len(self.images);
        parent  = [-1]      * len(self.images);
        dist    = [np.inf]  * len(self.images);
                        
        dist[src] = 0;        
        cur = src;
        while cur != dest:                           
            visited[cur] = True;
            
            # neighbors
            for i in range(len(self.images)):
                if visited[i] == False:
                    if self.match[cur][i] != None or self.match[i][cur] != None:
                        if dist[i] > dist[cur] + 1:
                            dist[i] = dist[cur] + 1;                    
                            parent[i] = cur;
            # next
            min = np.inf;            
            for i in range(len(self.images)):
                if visited[i] == False and dist[i] < min:
                    min = dist[i];
                    cur = i;     
                    
        # return the shortest path
        path = [];        
        while parent[cur] != -1:
            path.append(parent[cur]);
            cur = parent[cur];
        
        # in-place reverse
        path.reverse();
        return path;
    
    def stitch(self, ref = 0):
        """
        Stitch all images together.
        Input:
            ref: reference image index.
        """
        
        h, w = self.shape;        
        pylab.figure();
        pylab.ion();
        #pixels = np.ndarray(shape=(h, w, 3), dtype=float, order='C');
        pixels = np.zeros((h, w, 3));
        for i in range(h):
            print "Row: ", i
            for j in range(w):
                p = np.matrix([j, i, 1]).reshape((3, 1));
                for im in self.images:
                    # take the inverse homography to the current image's domain
                    q = np.ravel(im.H.I * p);
                    q[0] /= q[2];
                    q[1] /= q[2];
                    qh, qw = im.shape;
                    if q[0] < 0 or q[0] >= qw or q[1] < 0 or q[1] >= qh: continue;
                    color = im.interpolate(q[0], q[1]);
                    pixels[i, j] = color;
            
            #pylab.imshow(pixels);
            #pylab.draw();
        #return pixels;
        self.pixels = pixels;
        
    def show(self):
        #[im.show() for im in self.images];
        pylab.figure(0);
        self.match[0][1].show();
        
        pylab.figure(1);
        pylab.imshow(self.pixels);
        pylab.axis('image');
        
        pylab.show();
        

    def refinement(self):
        """
        Refine homography matrix H using Levenberg-Marquardt non-linear optimization.
        """
        # compute J'*J directly as J is sparse.
        
        
    
    def bundleAdjustment(self, match):
        """
        Every time add a new pair of images (ImageMatch) to the bundle and perform 
        refinement to all existing homography matrices.
        """
        self.bundle.append(match);
        
        self.refinement();
        
    def panaroma(self):
        """
        Produce panoramic images from homographies.
        """
        # compute the size of the panorama
        # for each pixel, do bilinear sampling
        # reference
        # http://www.ics.uci.edu/~dramanan/teaching/cs217_spring09/hw/hw3.html
        # Since the matching is at the integer pixel coordinates, here we also use 
        # integer pixel coordinates.
        # The origin of each image is located at the upper left.
        # Take image[0] as the reference image.
        stack = []
        
        # Use the residual in the paper but refine all H. Then render with new H and compare.
        
        
def main():
    imo = ImageMosaick();
    folder  = "./images";
    #images  = ["scene.pgm", "basmati.pgm"];
    #images = ["PICT0013.pgm", "PICT0014.pgm"];
    #images = ["PICT0014_800.pgm", "PICT0013_800.pgm"];
    images = ["PICT0015_800.pgm", "PICT0014_800.pgm"];
    imo.mosaick([folder + "/" + image for image in images]);
    imo.show();
    
if __name__ == "__main__":
    main();
        