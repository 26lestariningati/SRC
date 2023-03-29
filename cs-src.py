import numpy as np
from PIL import Image
import scipy.ndimage as spimg
import imageio as imgio
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time
import random
import warnings
from datetime import datetime
from time import strftime
from time import gmtime

warnings.filterwarnings("ignore")

#d = 80
SCALE = 1.0

j = 1

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def delta(x,i,class_num):
    '''
    Function that selects the coefficients associated with the ith class
    Useful for SCI calculation
    '''
    n,m = len(x),len(class_num)

    if (n != m):
        print('Vectors of differents sizes')

    tmp = i*np.ones(n)-class_num

    for k in range(n):
        if tmp[k]==0:
            tmp[k]=1
        else:
            tmp[k]=0

    return tmp*x

def residual(y,A,x,class_x):
    '''
    Returns the class which minimizes the reconstruction error following the norm 2
    '''
    k = np.max(class_x)+1
    r = np.zeros(k)

    for i in range(0,k):
        r[i] = np.linalg.norm(y - np.dot(A,delta(x,i,class_x)))

    return r

def SCI(x,class_num):
    '''
    - class_num: classe of a training element.
    - x        : sparse coefficients
    '''

    k = len(set(class_num)) # Number of different classes

    return (k*(1/np.linalg.norm(x,ord=1))*np.max([np.linalg.norm(delta(x,i,class_num),ord=1) for i in range(k)]) - 1)/(k-1)

def read_images(path, sz=None, sz0=120, sz1=165):
    '''
    Sizes must be changed depending on the pictures analyzed
    Resizing is possible with 'sz'
    '''
    X,y = [], []
    ext = [".jpg", ".pgm", ".png"]
    RGB_t = ["JPEG", "PNG"]
    Gray_t = ["PPM"]
    z = 0
    for dirname in np.sort(os.listdir(path)):
        if not os.path.isfile(path + dirname):
            for f_name in np.sort(os.listdir(path + dirname)):
                if f_name.endswith(tuple(ext)) and not f_name.startswith("."):
                    try:
                        im = Image.open(os.path.join(path + dirname + '/', f_name))
                        if im.format in RGB_t:
                            im = im.convert("L")
                        # resize to given size (if given) and check that it's the good size
                        #if ((im.size[0] == sz0) & (im.size[1]==sz1)):
                            #if (sz is not None):
                                #im = im.resize(sz, Image.NEAREST)
                        im = spimg.zoom(im,SCALE)
                        X.append(np.asarray(im, dtype=np.uint8))
                        c = int(dirname[1:3])
                        y.append(c)                        
                    except IOError:
                        pass
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
    '''
    for dirname , dirnames , filenames in os.walk(path):
        for filename in os.listdir(path):
            try:
                im = Image.open(os.path.join(path , filename))
                im = im.convert("L")
                # resize to given size (if given) and check that it's the good size
                if ((im.size[0] == sz0) & (im.size[1]==sz1)):
                    if (sz is not None):
                        im = im.resize(sz, Image.NEAREST)
                        X.append(np.asarray(im, dtype=np.uint8))
                        c = int(filename[2:5]) - 1
                        y.append(c)
            except IOError:
                pass
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
    '''

    print(path,"... Images uploaded !")
    return [X,y]

def stack(X_train,X_test):
    X_toconcat_train = [np.reshape(e,(X_train[0].shape[0]*X_train[0].shape[1],1)) for e in X_train]
    X_toconcat_test = [np.reshape(e,(X_test[0].shape[0]*X_test[0].shape[1],1)) for e in X_test]

    Xtrain = np.concatenate(X_toconcat_train,axis=1) # Each column is now an image of the train set
    Xtest = np.concatenate(X_toconcat_test,axis=1) # Each column is now an image of the test set

    return Xtrain,Xtest

def main(j):
    '''
    '''
    train_path_images = 'ORL/Train_Data/' # Pictures should be in here
    test_path_images = 'ORL/Test_Data/' # Pictures should be in here

    #X, y = read_images(train_path_images, sz=(30,42))
    _Xtrain, _ytrain = read_images(train_path_images)
    train_images_loaded = len(_Xtrain)
    train_class_loaded = len(list(set(_ytrain)))
    print("Number of pictures train loaded",train_images_loaded)
    print("Number of class train loaded",train_class_loaded)

    _Xtest, _ytest = read_images(test_path_images)
    test_images_loaded = len(_Xtest)
    test_class_loaded = len(list(set(_ytest)))
    print("Number of pictures test loaded",test_images_loaded)
    print("Number of class test loaded",test_class_loaded)

    X_train, X_test  = [], []
    ytrain, ytest = [], []
    indices_train, indices_test = [], []

    for i in range(len(_Xtrain)):
        X_train.append(_Xtrain[i])
        ytrain.append(_ytrain[i])
        indices_train.append(i)

    for i in range(len(_Xtest)):
        X_test.append(_Xtest[i])
        ytest.append(_ytest[i])
        indices_test.append(i)

    Xtrain, Xtest = stack(X_train,X_test)

    '''
    Performance test here
    '''
    from sklearn.linear_model import Lasso
    match = 0
    #test_count = len(ytest)
    start_data = 0
    end_data = 200
    start_i = 0
    print("Number of image test loaded", (end_data-start_data))

    # start input time measure module
    tickin = time.perf_counter()

    # Initial call to print 0% progress
    #printProgressBar(0, test_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
    #clf=classifier
    for i in range(start_data,(end_data-start_data)):
        #test_pic = i
        i = start_i + i
        clf = Lasso(alpha=4)
        y = Xtest[:,i]
        clf.fit(Xtrain,y)
        x = clf.coef_
        pred_class = np.argmin(residual(Xtest[:,i],Xtrain,x,ytrain))
        if ytest[i] == pred_class:
            match = match + 1
        # Update Progress Bar
        #printProgressBar(i + 1, test_count, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # start output time measure module
    tickout = time.perf_counter()

    print("Iteration: ", j)
    accuracy = float(match / (end_data-start_data)) * 100
    print("Accuracy: ", accuracy, "%")

    # display the result
    duration = strftime("%H:%M:%S",gmtime(tickout - tickin))
    #print(f"Process time: {tickout - tickin:0.4f} seconds")
    print(f"Process time: {duration} ")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main()
    else:
        try:
            while True:
                main(j)
                j = j + 1
        except KeyboardInterrupt:
            pass
