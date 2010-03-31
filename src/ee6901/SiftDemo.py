#------------------------------------------------------------------------------
# Son Hua, NUS
# 2010-03-23
# EE6901 Mosaicking
#------------------------------------------------------------------------------

#
# SIFT demo
#
import os
from numpy import *
from pylab import * # matlabplotlib
from sift import *

# the project should start with workspace_loc:PyMosaick
folder  = "./images";
#os.system("cd");
# does not affect current Python process. Use os.chdir instead.
#os.chdir(folder);
images  = ["scene.pgm", "box.pgm"];
keys    = ["scene.key", "box.key"];

# SIFT
process_image(folder + "/" + images[0], folder + "/" + keys[0]);
process_image(folder + "/" + images[1], folder + "/" + keys[1]);

locs = [[] for i in range(2)];
descriptors = [[] for i in range(2)];
locs[0], descriptors[0] = read_features_from_file(folder + "/" + keys[0]);
locs[1], descriptors[1] = read_features_from_file(folder + "/" + keys[1]);

scores = match(descriptors[0], descriptors[1]);
#os.chdir("..");

# plot
im = [[] for i in range(2)];
# flip the image so imshow and plot are consistent
im[0] = flipud(imread(folder + "/" + images[0])); 
im[1] = flipud(imread(folder + "/" + images[1]));

figure(0);
plot_features(im[0], locs[0]);
figure(1);
plot_features(im[1], locs[1]);
figure(2);
plot_matches(im[0], im[1], locs[0], locs[1], scores);
show();


