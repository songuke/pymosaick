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
import Image
import sys
import cProfile

class Liner(object):
    def __init__(self):
        pass
    def line(self, raster, src, dest, val):
        x0 = int(src[0]);
        y0 = int(src[1]);
        x1 = int(dest[0]);
        y1 = int(dest[1]);
        #height, width = np.shape(image);
        
        # Another Bresenham by me. This time ports to Python.    
        # now x0 < x1
        Dx = x1 - x0; # Dx >= 0 now
        Dy = y1 - y0;
        steep = (np.abs(Dy) >= np.abs(Dx));
        if steep:
            #SWAP(x0, y0);
            #SWAP(x1, y1);
    
            x0, y0 = y0, x0;
            x1, y1 = y1, x1;
            
            # recompute Dx, Dy after swap
            Dx = x1 - x0;
            Dy = y1 - y0;
            
        xstep = 1;
        if Dx < 0 :
            xstep = -1;
            Dx = -Dx;
            
        ystep = 1;
        if Dy < 0: # y1 < y0, decreasing
            ystep = -1;        
            Dy = -Dy; 
            
        TwoDy = 2*Dy; 
        TwoDyTwoDx = TwoDy - 2*Dx; # 2*Dy - 2*Dx
        E = TwoDy - Dx; #2*Dy - Dx
    
        y = y0;
        #int xDraw, yDraw;
        
        # FIXME: sometimes infinite loop here!? Cannot use <= or >= as don't know the line goes up or down.
        x = x0;
        
        #xDrawPrev = np.inf;
        #yDrawPrev = np.inf;
        while x != x1:
            if steep:            
                xDraw = y;
                yDraw = x;
            else:           
                xDraw = x;
                yDraw = y;
                
            # plot
            # avoid out of bound when stretching triangle
            #if (xDraw < 0 || xDraw >= width || yDraw < 0 || yDraw >= height) {}
            #else {
                #int index = yDraw * width + xDraw;    
                #image->setPixel(Float2(xDraw, yDraw), color);
            
            # trick for overlapping images -> mask values accumulate.
            #image[xDraw, yDraw] += val;
            
            # record only the beginning and end position of a line segment
            #if yDraw != yDrawPrev:
                #if yDrawPrev != np.inf:
                    # record last point of the previous segment
                    #raster.append((xDrawPrev, yDrawPrev, val));
                # record first point of the new segment
            raster.append((xDraw, yDraw, val)),
            
            #xDrawPrev = xDraw;
            #yDrawPrev = yDraw;
            
            
            # next
            if E > 0:
                E += TwoDyTwoDx; #E += 2*Dy - 2*Dx;
                y = y + ystep;
            else:
                E += TwoDy; #E += 2*Dy;
                        
            x += xstep;
        

class Feature(object):
    def __init__(self, imageId, descriptor, location):
        self.imageId    = imageId;
        self.descriptor = descriptor;
        self.location   = location;
        self.H          = None;
        self.mosaickShape = None;

#def featureDistance(f1, f2):
    #return 1.0 - abs(numpy.dot(f1.descriptor, f2.descriptor));
        
class ImageObject(object):
    '''
    ImageObject class encapsulates image pixel data, SIFT features, and 
    '''
    def __init__(self, id, pixels, locations, descriptors):
        self.id     = id;
        self.pixels = pixels;
        self.descriptors = descriptors;
        self.locations = locations;
        shape = np.shape(self.pixels);
        if len(shape) == 2:
            self.shape = shape;
            self.channels = 1;
        else:
            self.shape = (shape[0], shape[1]);
            self.channels = shape[2];
        #self.center = (self.shape[1] * 0.5, self.shape[0] * 0.5);
        
        # precompute center weight for every pixel
        h, w = self.shape;
        h2, w2 = 1.0 / h, 1.0 / w;
        weight = np.zeros((h, w));
        cx = 0.5; cy = 0.5;
        sigma2 = 0.25**2;
        for i in range(h):
            for j in range(w):
                weight[i, j] = np.exp(-((i * h2 - cy)**2 + (j * w2 - cx)**2) / sigma2); 
                #weight[i, j] = (0.5 - np.abs(i * h2 - cy)) * (0.5 - np.abs(j * w2 - cx)); # abs is too slow.
                #weight[i, j] = 0.25 - np.abs((i * h2 - cy) * (j * w2 - cx));
        self.weight = weight;
        
    def interpolate(self, x, y):
        #h, w = self.shape;
        
        x0 = np.floor(x);
        x1 = x0 + 1;
        y0 = np.floor(y);
        y1 = y0 + 1;
        
        """
        if x0 < 0:  x0 = 0;
        if x0 >= w: x0 = w - 1;
        if x1 < 0:  x1 = 0;
        if x1 >= w: x1 = w - 1;
        if y0 < 0:  y0 = 0;
        if y0 >= h: y0 = h - 1;
        if y1 < 0:  y1 = 0;
        if y1 >= h: y1 = h - 1;
        """
        
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
        
        # dictionary to map ID to ImageObject object
        self.totalImages    = 0;
        self.dictIdImage    = {};
        self.dictFileImage  = {};
        
    # Load image from a file and return ImageObject object
    def loadImage(self, file):
        if file in self.dictFileImage:
            return self.dictFileImage[file];
        else:
            id     = self.totalImages;
            pixels = pylab.flipud(pylab.imread(file));            
            
            # SIFT features
            partName, partDot, partExt = file.rpartition('.');
            keyFile = ''.join(partName + partDot + "key"); 
            pgmFile = ''.join(partName + partDot + "pgm");
            if os.path.exists(pgmFile) == False:
                #pylab.imsave(pgmFile, pixels);
                if len(pixels.shape) == 2:
                    pilImage = Image.fromarray(pixels, 'L');                    
                else:
                    h = pixels.shape[0];
                    w = pixels.shape[1];                    
                    pixelsGray = np.matrix(np.zeros((h, w)), dtype=np.uint8);                    
                    for i in range(h):
                        for j in range(w):
                            pixelsGray[i, j] = (np.mean(pixels[i, j])).astype(np.uint8);
                    pilImage = Image.fromarray(pixelsGray, 'L');
                pilImage.save(pgmFile);
                
            if os.path.exists(keyFile) == False:            
                sift.process_image(pgmFile, keyFile);
                
            loc, des  = sift.read_features_from_file(keyFile);
            #im.features = [Feature(im.id, des[i], loc[i]) for i in range(len(des))];
            #print "Total features: ", len(im.features)
            
            im = ImageObject(id, pixels, loc, des);
            
            # add to dictionary
            self.dictFileImage[file]    = im;
            self.dictIdImage[im.id]     = im;
            # increase total images
            self.totalImages += 1;
            #print "Total images: ", self.totalImages;
            return im;
    
    def getImageByID(self, id):
        if id in self.dictIdImage:
            return self.dictIdImage[id];
        else:
            return None;
        
class ImageMatch(object):
    def __init__(self, image1, image2):
        self.image1 = image1;
        self.image2 = image2;
        self.locs1 = [];
        self.locs2 = [];
        
    def __homography__(self, idx):
        """
        Compute homography matrix to transform image 2 to image 1.
        idx: index to a subset of correspondences for homography computation.
        Return a 3x3 matrix [h11, h12, h13; h21, h22, h23; h31, h32, h33].
        
        Location from SIFT key file is in (y, x) format.
        """
        # each correspondence gives two equations
        nbEqns = len(idx) * 2;
        A = np.matrix(np.zeros((nbEqns, 9)));
        #b = np.zeros((nbEqns, 1));
        for i in range(len(idx)):
            p1 = self.locs1[idx[i]]; p1x = p1[1]; p1y = p1[0];
            p2 = self.locs2[idx[i]]; p2x = p2[1]; p2y = p2[0];
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
        bestMask = None;
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
            if bestMask == None or newInliers > maxInliers:
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
        inliers = np.sum(bestMask);
        outliers = len(bestMask) - np.sum(bestMask);
        print "Best homography: ", H
        print "Inliers/Outliers: ", inliers, "/", outliers;
        # store homography matrix
        self.H = H;
        return inliers, outliers 
                    
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
            p2 = np.matrix([ [self.locs2[i][1]], [self.locs2[i][0]], [1] ]);
            p1 = np.ravel(H * p2);
            p1 /= p1[2]; # convert back to non-homogeneous
            
            # compute the distance of p1 and original locs1
            # NOTE: p1[0] is x, self.locs1[i][1] is x. The way of index is quite confusing here!
            dist = (p1[0] - self.locs1[i][1])**2 + (p1[1] - self.locs1[i][0])**2;
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
        
    def mosaick(self, imageFiles, ref=-1):
        start = time.clock();
        
        manager = ImageManager();
        self.images = [manager.loadImage(file) for file in imageFiles];
        print "Total images: ", manager.totalImages;
        
        elapsed = time.clock() - start;
        print "SIFT: ", elapsed; 
        
        start = time.clock();
        self.findCorrespondenceKdTree();
        #self.findCorrespondenceBruteForce();
        elapsed = time.clock() - start;
        print "Kdtree: ", elapsed; 
        
        # print the number of correspondences for each match before RANSAC
        # remove match that has two low correspondences
        
        low = [];
        for m in self.match:
            nbCorrs = len(self.match[m].locs1);
            print m, " has ", nbCorrs, " correspondences."            
            if nbCorrs < 4:
                low.append(m);
                #print m, " has too low correspondences and is discarded."
            
        for m in low:
            del self.match[m];
        
        # recover H for each pair of images
        # and use RANSAC to reject outliers
        start = time.clock();
        """
        for i in range(len(self.images) - 1):
            for j in range(i + 1, len(self.images)): 
                inliers, outliers = self.match[i][j].ransac();
                # remove incorrect matches
                #if outliers > 1.5 * inliers:
                #    self.match[i][j] = None;
        """
        low = [];
        for m in self.match:
            inliers, outliers = self.match[m].ransac();
            if inliers < 15 or inliers < 0.1 * outliers:
                low.append(m);
        for m in low:
            del self.match[m];
            
        elapsed = time.clock() - start;
        print "RANSAC: ", elapsed;
                
        # find the reference image so that the global transform produces
        # the smallest area
        if ref == -1:
            minArea = np.inf;
            minRef = -1;
            for i in range(len(self.images)):
                gh, gw = self.findGlobalTransform(ref=i);
                area = gh * gw;
                if area < minArea:
                    minArea = area;
                    minRef = i;        
            ref = minRef;
            print "Automatic reference image: ", minRef;
        else:
            print "Warning: Manual reference image can result in bad mosaick. Out of memory may occur.";
        
        # find global transform to reference images        
        start = time.clock();
        self.shape = self.findGlobalTransform(ref); 
        elapsed = time.clock() - start;
        print "Global transform: ", elapsed;
                
        # perform stitching
        start = time.clock();
        self.stitch(ref);
        elapsed = time.clock() - start;
        print "Stitch: ", elapsed; 
        
        # save to disk
        dir = './mosaick';        
        try:
            os.makedirs(dir);
        except OSError:
            pass

        for i in range(1000):
            mosaickFile = dir + "/" + "Mosaick%04d.png" % i;
            if os.path.exists(mosaickFile) == False: break;    
             
        #pylab.imsave(mosaickFile, self.pixels);       
        pilImage = Image.fromarray(self.pixels);
        pilImage.save(mosaickFile);
        print "Saved to ", mosaickFile;
        
    def findCorrespondenceBruteForce(self):
        """
        Find correspondence using brute-force scan.
        """
        self.match = {};
        distRatio = 0.8;
        for i in range(len(self.images) - 1):
            im = self.images[i];
            for j in range(i + 1, len(self.images)):
                jm = self.images[j];
                # for every feature in image[i]
                for m in range(len(im.descriptors)):
                    f = im.descriptors[m];
                    nearestDist = np.inf;
                    secondNearestDist = np.inf;
                    nearestIndex = -1;
                    # compare with every feature in image[j]
                    for n in range(len(jm.descriptors)):
                        g = jm.descriptors[n];
                        dist = KdtreeCustom.minkowski_distance(f, g, 2);
                        if dist < nearestDist:
                            nearestDist = dist;
                            nearestIndex = n;
                        if dist > nearestDist and dist < secondNearestDist:
                            secondNearestDist = dist;
                    if nearestDist < distRatio * secondNearestDist:
                        # match
                        if (i, j) not in self.match:     
                            self.match[(i, j)] = ImageMatch(im, jm);
                        immatch = self.match[(i, j)];                                   
                        immatch.locs1.append(im.locations[m]);
                        immatch.locs2.append(jm.locations[nearestIndex]);
                        
    def findCorrespondenceKdTree(self):
        """
        Find coresspondence using KdTree.
        """
        # Note: cannot use below statement to create a 2d list.
        # The rows are duplicated! Objects from the second row are the same as 
        # the first row!
        #self.match = [[None] * len(self.images)] * len(self.images);
        # Use dictionary instead.
        self.match = {};
        """
        for i in range(len(self.images) - 1):
            for j in range(i + 1, len(self.images)):
                self.match[i][j] = ImageMatch();
                self.match[i][j].image1 = self.images[i];
                self.match[i][j].image2 = self.images[j];
        """
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
                #dist, idx = kdtree.query(ft, knearest, eps=0.01, distance_upper_bound=0.1); # prune more? 
                dist, idx = kdtree.query(ft, knearest); # prune more?
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
                                if i < m:                                    
                                    if self.match.has_key((i, m)) == False:
                                        immatch = ImageMatch(self.images[i], self.images[m]);
                                        self.match[(i, m)] = immatch;                                        
                                    self.match[(i, m)].locs1.append(self.images[i].locations[n]);
                                    self.match[(i, m)].locs2.append(locations[idx[j]]);
                                else:
                                    if self.match.has_key((m, i)) == False:
                                        immatch = ImageMatch(self.images[m], self.images[i]);
                                        self.match[(m, i)] = immatch;
                                    self.match[(m, i)].locs2.append(self.images[i].locations[n]);
                                    self.match[(m, i)].locs1.append(locations[idx[j]]);
                """
                else:
                    if dist < 0.35:
                        m = imageIds[idx];
                        self.match[i][m].locs1.append(self.images[i].locations[n]);
                        self.match[i][m].locs2.append(locations[idx]);
                """
                     
                #elapsed = time.clock() - start;
                #print "Remaining: ", elapsed  
            #print match[n];
            #print [imageIds[match[n][j]] for j in range(knearest)];
        
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
            h, w = im.shape;
            xmin = 0; xmax = w - 1;
            ymin = 0; ymax = h - 1;
                        
            im.corners = [np.matrix([xmin, ymin, 1]).reshape(3, 1), 
                       np.matrix([xmax, ymin, 1]).reshape(3, 1), 
                       np.matrix([xmax, ymax, 1]).reshape(3, 1), 
                       np.matrix([xmin, ymax, 1]).reshape(3, 1)
                       ];
            if im.id == ref: continue;
            
            # find a path to imageRef
            path = self.findPath(im.id, ref);
            # accumulate H
            A = np.matrix(np.eye(3));
            
            #h = 0; w = 0;
            #xmin = np.inf; xmax = -np.inf;
            #ymin = np.inf; ymax = -np.inf;
            #if len(path) > 0:
                #src = path[0];
                #imSrc = self.images[src];
                #h, w = imSrc.shape;
                
            for j in range(len(path) - 1):
                # project path[j] to path[j+1]
                src = path[j];
                dest = path[j+1];
                
                if dest < src:
                    H = self.match[(dest, src)].H;
                else:
                    # inverse
                    H = self.match[(src, dest)].H.I;
                
                #imSrc = self.images[src];
                #imDest = self.images[dest];
                    
                # find xmin and ymin
                #if h == 0:
                #    h, w = imSrc.shape;            
                    
                #hj, wj = imDest.shape;
                # project to image[j]
                #corners = np.matrix([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]]);
                #xmin = 0; xmax = wj - 1;
                #ymin = 0; ymax = hj - 1;
                
                for c in range(len(im.corners)):
                    """
                    p = np.matrix(c.reshape((3, 1)));
                    q = np.ravel(H * p);
                    q /= q[2];
                    xmin = np.amin([xmin, q[0]]);
                    ymin = np.amin([ymin, q[1]]);
                    xmax = np.amax([xmax, q[0]]);
                    ymax = np.amax([ymax, q[1]]);
                    """
                    im.corners[c] = H * im.corners[c];
                
                # update the new image size
                #w = np.ceil(xmax - xmin);
                #h = np.ceil(ymax - ymin);
                
                # translate to new origin (xmin, ymin)
                #T = np.matrix([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]]);
                #A = A * H * T;
                A = H * A;
                
            # store global homography for each image
            im.H = A;
            #im.shape = (h, w);
            
            # save the four corners
            #if len(path) > 0:
            for c in im.corners:
                #c = np.ravel(c);
                c /= c[2];
                gxmin = np.amin([gxmin, c[0]]);
                gymin = np.amin([gymin, c[1]]);
                gxmax = np.amax([gxmax, c[0]]);
                gymax = np.amax([gymax, c[1]]);
                    
            # find global image size
            """
            gxmin = np.amin([gxmin, xmin]);
            gymin = np.amin([gymin, ymin]);
            gxmax = np.amax([gxmax, xmax]);
            gymax = np.amax([gymax, ymax]);
            """
        
        # return the global size
        gw = np.ceil(gxmax - gxmin);
        gh = np.ceil(gymax - gymin);
        T = np.matrix([[1, 0, -gxmin], [0, 1, -gymin], [0, 0, 1]]);
        for im in self.images:
            im.H = T * im.H; 
            for c in im.corners:
                c += np.matrix([[-gxmin], [-gymin], [0]]);
        
        #self.shape = (gh, gw);
        return (int(gh), int(gw));

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
                    if self.match.has_key((cur, i)) or self.match.has_key((i, cur)):
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
        path = [dest];
        cur = dest;        
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
        print "Mosaick size: (width = %d, height = %d)" % (w, h);
        
        # precompute all inverse matrices
        for im in self.images:
            im.HI = np.array(im.H.I);
        
        # draw border lines of images to mask        
        liner = Liner();
        raster = [];
        for im in self.images:
            val = im.id;
            for c in range(len(im.corners)):
                d = (c + 1) % len(im.corners);
                liner.line(raster,  (im.corners[c][0], im.corners[c][1]), 
                                    (im.corners[d][0], im.corners[d][1]), val);
        # only keep the min and max x of every (y, val) pair
        rasterMinMax = {};
        for x, y, v in raster:
            if (y, v) not in rasterMinMax:
                rasterMinMax[(y, v)] = [np.inf, -np.inf]; # [min, max]
            mm = rasterMinMax[(y, v)];
            if x < mm[0]:
                mm[0] = x;
            if x > mm[1]:
                mm[1] = x;
            
        # for every (x, y) show a list of images at this position
        rasterDictY = {};
        for y, v in rasterMinMax:
            if y not in rasterDictY:
                rasterDictY[y] = {};
            rasterDictX = rasterDictY[y];
            for x in rasterMinMax[(y, v)]:      
                if x not in rasterDictX: 
                    rasterDictX[x] = set(); # use 
                l = rasterDictX[x];
                if v not in l:
                    l.add(v);

        #pylab.figure();
        #pylab.ion();
        #pixels = np.ndarray(shape=(h, w, 3), dtype=float, order='C');
        # pixels are stored in uint8 data type so save storage and increase performance.
        if self.images[0].channels > 1:
            pixels = np.zeros((h, w, self.images[0].channels), dtype=np.uint8);
        else:
            pixels = np.zeros((h, w), dtype=np.uint8);
            
        # stitching
        zero = [0] * self.images[0].channels;
        false = [False] * len(self.images);
        
        for i in range(h):
            #print "Row: %d/%d" % (i, h-1)
            # keep track of the current overlap images
            #overlap = [];
            if i not in rasterDictY: continue;
            
            activeImages = false; # all false
            rasterDictY_i = rasterDictY[i];
            for j in range(w):
                # update overlap images
                #rasterList = [];
                #if i in rasterDictY:
                #if not rasterDictY_i is None:
                #if j in rasterDictY[i]:
                if j in rasterDictY_i:
                    #rasterList = rasterDictY[i][j];
                    rasterList = rasterDictY_i[j];
                
                    #removeList = [];
                    for r in rasterList:
                        """
                        try:
                            #index = overlap.index(r);
                            overlap.index(r);
                            # current image is already found in overlap
                            # this means we are going out of this image. After this loop remove this image.
                            removeList.append(r);
                        except ValueError:
                            # this image is not found in overlap.
                            # Add this image and start using it.
                            overlap.append(r);
                        """
                        
                        """
                        found = False;
                        for o in overlap:
                            if o == r:
                                removeList.append(r);
                                found = True;
                                break;
                        if not found:
                            overlap.append(r);
                        """
                        # ignore the last pixel (on boundary).
                        activeImages[r] = not activeImages[r];                        
                    
                #p = np.matrix([[j], [i], [1]]);
                p = [j, i, 1];
                sum = zero;
                total = 0;
                #for im in self.images:
                #for o in overlap:
                for o in range(len(self.images)):
                    if not activeImages[o]: continue;
                     
                    im = self.images[o];
                    
                    # take the inverse homography to the current image's domain
                    # Note: inverse matrix H every time causes a lot of 
                    # performance penalty.
                    #q = np.ravel(im.H.I * p);
                    #q = np.ravel(im.HI * p);
                    # Note: doing ravel and np.matrix multiplication is inconvenient
                    # and incurred 3 times performance penalty as compared to 
                    # np.dot and list.
                    q = np.dot(im.HI, p);
                    q /= q[2];
                    qh, qw = im.shape;
                    if q[0] < 0 or q[0] >= qw or q[1] < 0 or q[1] >= qh: continue;
                    # no interpolation at boundary
                    if (qw - 1 <= q[0] and q[0] < qw) or (qh - 1 <= q[1] and q[1] < qh): 
                        color = im.pixels[np.floor(q[1]), np.floor(q[0])];
                    else:
                        color = im.interpolate(q[0], q[1]);
                    
                    # weight
                    #w = (q[1] - im.center[0])**2 + (q[0] - im.center[1])**2;
                    we = im.weight[np.floor(q[1]), np.floor(q[0])];
                    sum += we * color;
                    total += we;
                if total > 0:
                    pixels[i, j] = (sum / total).astype(np.uint8);
                    
                # clean up any complete overlap image
                #for r in removeList:
                #    overlap.remove(r);
                    
        self.pixels = pixels;
        
    def show(self):
        #[im.show() for im in self.images];
        
        """        
        print range(len(self.images) - 1)
        # show each image match pair
        for i in range(len(self.images) - 1):
            for j in range(i + 1, len(self.images)):
                print "Figure (%d, %d)" % (i, j)
        """
        for m in self.match: 
            pylab.figure();
            self.match[m].show();
                
        # show final mosaick
        pylab.figure();
        pylab.imshow(self.pixels);
        pylab.axis('image');
        
        pylab.show();
        
    '''
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
    '''
        
def main():
    folder  = "./images";
    if len(sys.argv) < 2:
        print "Usage: mosaick.exe <image set file>"
        return;
    
    file = sys.argv[1];
    
    # image set folder
    partSetFolder, partDot, partExt = file.rpartition('.');
    partPath, partSlash, setFolder = partSetFolder.rpartition('/');
    
    # read image list from file
    f = open(file, 'r');
    images = f.readlines();
    f.close();
    
    # remove any empty lines
    removeList = [];
    for i in range(len(images)):
        # check for existence
        # remove empty lines
        images[i] = images[i].strip();
        if images[i] == "":
            removeList.append(i);
    
    removeList.reverse();
    for r in removeList:
        del images[r];
    
    # full path
    fullImages = [folder + "/" + setFolder + "/" + image for image in images];    
    # print out image list
    print "Input images: "
    for i in range(len(images)):
        if not os.path.exists(fullImages[i]):
            print images[i], " not found!"
        else:
            print images[i];
    
    # start mosaicking
    imo = ImageMosaick();
    imo.mosaick(fullImages);
    #imo.show();
    
    """
    OpenGL to show the process interactively.
    """
    
if __name__ == "__main__":
    main();
    #cProfile.run('main()')
    