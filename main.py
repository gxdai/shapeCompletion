import argparse
import os 
import scipy.misc 
import numpy as np
import os

from model import shapeCompletion
#from model_ce import shapeCompletion            # The model is borrowed from context encoding
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--logdir', dest='logdir', default='./logs', help='name of the dataset')
parser.add_argument('--imageRootDir', dest='imageRootDir', default=None, help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=64, help='then crop to this size')
parser.add_argument('--output_size', dest='output_size', type=int, default=64, help='then crop to this size')
parser.add_argument('--shape_size', dest='shape_size', type=int, default=64, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--num_gpus', dest='num_gpus', type=int, default=3, help='# of gpus')

parser.add_argument('--train_list', dest='train_list', default='./modelnet10/script/train.txt', help='name of the dataset')
parser.add_argument('--test_list', dest='test_list', default='./modelnet10/script/test.txt', help='name of the dataset')

args = parser.parse_args()

print("args.imageRootDir = {:10}".format(args.imageRootDir))
print("{:10} = {:4d}".format('args.num_gpus', args.num_gpus))

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        print(args.checkpoint_dir)
        model = shapeCompletion(sess, shape_size=args.shape_size, batch_size=args.batch_size,
                        output_size=args.output_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir, 
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc,
                        train_list=args.train_list, test_list=args.test_list, logdir=args.logdir)
                        #imageRootDir=args.imageRootDir)

        if args.phase == 'train':
            model.train(args)
            #model.train_multiple_gpu(args)
        else:
            model.test(args)


"""
def main(_):
    model = shapeCompletion()
    shape = tf.placeholder(tf.float32, shape=[12, 32, 32, 32, 1], name='input')
    out = model.generator(shape)
    prob = model.discriminator(out)
    var_list = tf.trainable_variables()
    for var in var_list:
        print(var.name)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(10):
                prob_, out_ = sess.run([prob, out], feed_dict={shape: np.random.random((12, 32, 32, 32, 1))})
                print(out_.shape)
                print(prob_[0].shape)


"""






if __name__ == '__main__':
    tf.app.run()
