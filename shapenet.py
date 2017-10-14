import numpy as np
import os
from random import shuffle
import argparse
import sys
class Dataset:
    def __init__(self, train_listFile, test_listFile, test_benchmark=None, voxel_size=32, truncation=3, fileRootDir=None):
        # The format of input list is as follows:
        #
        #   *.sdf   *df     label
        #   *.sdf   *df     label
        #   *.sdf   *df     label
        #       ...
        ###################################################################
        with open(train_listFile) as f:         # The training list file
            self.train_list = f.readlines()
        with open(test_listFile) as f:          # The testing list file
            self.test_list = f.readlines()
        with open(test_benchmark) as f:
            self.benchmark_list = f.readlines()


        def getFullPath(fileRootDir, relativePathList):
            # Get the path list of data
            data = [os.path.join(fileRootDir, tmp.split(' ')[0]) for tmp in relativePathList]
            # Get the path list of target 
            target = [os.path.join(fileRootDir, tmp.split(' ')[1]) for tmp in relativePathList]

            return data, target

        # Get the path list of training data
        self.train_data, self.train_target = getFullPath(fileRootDir, self.train_list) 
        # Get the path list of testing data
        self.test_data, self.test_target = getFullPath(fileRootDir, self.test_list) 
        # Get the path list of benchmark data
        self.benchmark_list = [os.path.join(fileRootDir, 'shapenet_dim32_sdf', f.rstrip('\n')+'__0__.sdf') for f in self.benchmark_list]

        
        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.benchmark_ptr = 0
        self.train_size = len(self.train_list)
        self.test_size = len(self.test_list)
        self.benchmark_size = len(self.benchmark_list)
	self.voxel_size = voxel_size
        self.truncation = truncation


    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_data[self.train_ptr: self.train_ptr+batch_size]
                labels = self.train_target[self.train_ptr: self.train_ptr+batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr+batch_size) % self.train_size

                paths = self.train_data[self.train_ptr:] + self.train_data[:new_ptr]
                labels = self.train_target[self.train_ptr:] + self.train_target[:new_ptr]
                self.train_ptr = new_ptr
                # shuffle order at the beginning of every new epoch of trainning
                tmp = zip(self.train_data, self.train_target)
                shuffle(tmp)
                self.train_data, self.train_target  = zip(*tmp)         # shuffle the list
                self.train_data = list(self.train_data) 
                self.train_target = list(self.train_target)


        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_data[self.test_ptr: self.test_ptr+batch_size]
                labels = self.test_target[self.test_ptr: self.test_ptr+batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr+batch_size) % self.test_size
                paths = self.test_data[self.test_ptr:] + self.test_data[:new_ptr]
                labels = self.test_target[self.test_ptr:] + self.test_target[:new_ptr]
                self.test_ptr = new_ptr
                # shuffle order at the beginning of every new epoch of testning

                self.test_data, self.test_target  = zip(*shuffle(zip(self.test_data, self.test_target)))         # shuffle the list
                self.test_data = list(self.test_data) 
                self.test_target = list(self.test_target)

        elif phase == 'random':
            # Pick random samples for testing
            random_index = np.random.permutation(self.test_size)
            paths = []
            labels = []
            for i in range(batch_size):
                paths.append(self.test_data[random_index[i]])
                labels.append(self.test_target[random_index[i]])
        
        elif phase == 'evaluation':
            if self.benchmark_ptr + batch_size < self.benchmark_size:
                paths = self.benchmark_list[self.benchmark_ptr: self.benchmark_ptr+batch_size]
                labels = self.benchmark_list[self.benchmark_ptr: self.benchmark_ptr+batch_size]
                self.benchmark_ptr += batch_size
            else:
                new_ptr = (self.benchmark_ptr+batch_size) % self.benchmark_size
                paths = self.benchmark_list[self.benchmark_ptr:] + self.benchmark_list[:new_ptr]
                labels = self.benchmark_list[self.benchmark_ptr:] + self.benchmark_list[:new_ptr]
                self.benchmark_ptr = new_ptr
                # shuffle order at the beginning of every new epoch of trainning
        else:
            return None, None
        # put parital shape and complete shape in the same numpy array
        # The first channel is partial shape
        # The second channel is complete shape
        voxels = np.ndarray([batch_size, self.voxel_size, self.voxel_size, self.voxel_size, 2])
        # masks: The L1 loss should only consider those (*.sdf < -1)
        masks = np.zeros((batch_size, self.voxel_size, self.voxel_size, self.voxel_size, 1))
        for i in xrange(len(paths)):
            # Load data (.txt)
            data = np.loadtxt(paths[i])
            data = np.reshape(data, (self.voxel_size, self.voxel_size, self.voxel_size, 1))
            # Load target (.txt)
            target = np.loadtxt(labels[i])
            target = np.reshape(target, (self.voxel_size, self.voxel_size, self.voxel_size, 1))

            # set mask label of unknown part to 1., the mask label of known part is 0
            voxels[i, :, :, :, :1] = data
            voxels[i, :, :, :, 1:] = target

        # Get the masks
        index = np.where(voxels[:, :, :, :, :1] < -1.)
        masks[index] = 1.       # Pick the unknown part for L1 loss
        if self.truncation:
            voxels = np.absolute(voxels)
            index = np.where(voxels > self.truncation)
            voxels[index] = self.truncation

        return voxels, masks      # Pick the cropped region for reconstruction loss
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is for loading shapenet partial data')
    parser.add_argument('--train_listFile', type=str, default='./data/h5_shapenet_dim32_sdf/train_file_label.txt', help='The training list file')
    parser.add_argument('--test_listFile', type=str, default='./data/h5_shapenet_dim32_sdf/test_file_label.txt', help='The testing list file')
    parser.add_argument('--fileRootDir', type=str, default='/home/gxdai/MMVC_LARGE/Guoxian_Dai/data/shapeCompletion/txt', help='The root direcoty of input data')
    parser.add_argument('--voxel_size', type=int, default=32, help='The size of input voxel')
    parser.add_argument('--truncation', type=float, default=3, help='The truncation threshold of input voxel')
    args = parser.parse_args()

    data = Dataset(train_listFile=args.train_listFile, test_listFile=args.test_listFile, fileRootDir=args.fileRootDir,\
            voxel_size=args.voxel_size, truncation=args.truncation)
    for _ in range(10):
        voxels, masks = data.next_batch(12, 'train')
        print(np.amax(voxels))
        print(np.amin(voxels))
        print(np.amax(masks))
        print(np.amin(masks))
        print("**")
    print("DONE")
"""
