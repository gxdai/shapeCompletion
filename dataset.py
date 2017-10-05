import numpy as np
import scipy.io as sio
from random import shuffle
class Dataset:
    def __init__(self, train_list, test_list, voxel_size):
        # Loding training voxel and label
        with open(train_list) as f:
            lines = f.readlines()
            # Shuffle data
            shuffle(lines)
            self.train_voxel = []
            self.train_label = []
            for l in lines:
                items = l.split()
                self.train_voxel.append(items[0])
                self.train_label.append(items[1])
        # Loading testing voxel and label
        with open(test_list) as f:
            lines = f.readlines()
            # Shuffle data
            shuffle(lines)
            self.test_voxel = []
            self.test_label = []
            for l in lines:
                items = l.split()
                self.test_voxel.append(items[0])
                self.test_label.append(items[1])

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
	self.voxel_size = voxel_size
        self.n_classes = 10
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_voxel[self.train_ptr: self.train_ptr+batch_size]
                labels = self.train_label[self.train_ptr: self.train_ptr+batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr+batch_size) % self.train_size
                paths = self.train_voxel[self.train_ptr:] + self.train_voxel[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
                # shuffle order at the beginning of every new epoch of trainning
                new_order = np.random.permutation(self.train_size)
                new_order = list(new_order)
                temp_voxel = [self.train_voxel[i] for i in new_order]
                temp_label = [self.train_label[i] for i in new_order]
                self.train_voxel = temp_voxel
                self.train_label = temp_label
                ### end of shuffling ################################
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_voxel[self.test_ptr: self.test_ptr+batch_size]
                labels = self.test_label[self.test_ptr: self.test_ptr+batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr+batch_size) % self.test_size
                paths = self.test_voxel[self.test_ptr:] + self.test_voxel[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        elif phase == 'random':
            # Pick random samples for testing
            random_index = np.random.permutation(self.test_size)
            paths = []
            labels = []
            for i in range(batch_size):
                paths.append(self.test_voxel[random_index[i]])
                labels.append(self.test_label[random_index[i]])
        else:
            return None, None
        # put parital shape and complete shape in the same numpy array
        # The first channel is partial shape
        # The second channel is complete shape
        voxels = np.ndarray([batch_size, self.voxel_size, self.voxel_size, self.voxel_size, 2])
        masks = np.ndarray([batch_size, self.voxel_size, self.voxel_size, self.voxel_size, 1])
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        
        for i in xrange(len(paths)):
            # Load matlab data
            # print("The testing model is {:20}".format(paths[i]))
            dataVol = sio.loadmat(paths[i])
            temp_data = np.reshape(dataVol['instance'], (self.voxel_size, self.voxel_size, self.voxel_size, 1))
            # Convert voxel data from [0, 1] to [-1, 1]
            temp_data = 2. * temp_data - 1.0
            # Create a random MASK
            """
            mask = np.random.random((self.voxel_size, self.voxel_size, self.voxel_size, 1))
            # randomly remove 30% voxels
            mask = mask < 0.7
            """
            mask = self.cropHole()          # The label of copped region is 0, while the label of the remaining region is 1
            partial_shape = np.multiply(mask.astype(np.float32), temp_data)
            voxels[i, :, :, :, :1] = partial_shape
            voxels[i, :, :, :, 1:] = temp_data
            masks[i, :, :, :, :] = mask
            one_hot_labels[i][int(labels[i])] = 1
        
        return voxels, 1-masks      # Pick the cropped region for reconstruction loss
    def cropHole(self):
        # center range (16, 48)
        # radius range (5, 15)
        rand_center = np.random.permutation(np.arange(5, 60))[:3]        # pick the first three numbers as center for croping
        rand_radius = np.random.permutation(np.arange(5, 30))[:3]                                 # randomly crop half the size
        rand_left = np.maximum(rand_center-rand_radius, 0)
        rand_right = np.minimum(rand_center+rand_radius, 64)

        # The cubic is 
        mask = np.ones((self.voxel_size, self.voxel_size, self.voxel_size, 1))
        mask[rand_left[0]:rand_right[0],
                rand_left[1]:rand_right[1],
                rand_left[2]:rand_right[2]] = 0
        return mask




        


