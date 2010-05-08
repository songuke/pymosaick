1. sift.py implementation is by Jan Erik Solem and available online at:
http://www.janeriksolem.net/2009/02/sift-python-implementation.html

I fixed some bugs in sift.py and adapted it for use with ImageMosaick.

sift.py basically calls siftWin32.exe to detect feature points. siftWin32.exe is
provided on SIFT's author's website.

2. ImageMosaick.py: main file of image mosaick implementation.

3. Please copy the siftWin32.exe to the code's folder. siftWin32.exe is available at:
http://www.cs.ubc.ca/~lowe/keypoints/siftDemoV4.zip

4. Install Python 2.6.5 and the following libraries.
Python 2.6.5
http://www.python.org/ftp/python/2.6.5/python-2.6.5.msi
Python Imaging Library 
http://effbot.org/downloads/PIL-1.1.7.win32-py2.6.exe
NumPy
http://sourceforge.net/projects/numpy/files/NumPy/1.4.1rc3/numpy-1.4.1rc3-win32-superpack-python2.6.exe/download
SciPy
http://sourceforge.net/projects/scipy/files/scipy/0.7.2rc3/scipy-0.7.2rc3-win32-superpack-python2.6.exe/download
Matplotlib
http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.99.1/matplotlib-0.99.1.win32-py2.6.exe/download

Make sure python.exe is can be found by PATH environment variables. If not, in command window, type: set PATH=%PATH%;C:\Python26; (assume your Python is installed at C:\Python26).

5. Run ImageMosaick in the command window.
python ImageMosaick.py <imageset> to run.

For example,
python ImageMosaick.py images/field2.txt

Binh Son.


