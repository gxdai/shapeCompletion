from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange 
from ops import *
from utils import *
import sys 
#from dataset import Dataset 
from shapenet import Dataset

class shapeCompletion(Dataset):
    def __init__(self, sess, shape_size=64,
                 batch_size=12, output_size=64, gf_dim=16, df_dim=16, L1_lambda=100,
                 input_c_dim=1, output_c_dim=1, dataset_name='modelnet10',
                 checkpoint_dir=None, sample_dir=None, train_list=None, test_list=None, logdir=None, train_listFile=None, test_listFile=None,
                 fileRootDir=None, truncation=3):
    
        # For dataset class
        self.sess = sess
        self.shape_size = shape_size
        self.batch_size = batch_size 
        self.output_size = output_size 


        self.checkpoint_dir = checkpoint_dir
        self.dataset_name = dataset_name
        self.logdir = logdir

        self.train_list = train_list
        self.test_list = test_list
        self.voxel_size = shape_size
        Dataset.__init__(self, train_listFile=train_listFile, test_listFile=test_listFile, fileRootDir=fileRootDir,\
                    voxel_size=shape_size, truncation=truncation)

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.output_c_dim = output_c_dim
        self.input_c_dim = input_c_dim

        self.L1_lambda = L1_lambda
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')

        """
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')
        """

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')

        """
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        """
        self.build_model(withLocalLoss=False)
 
 
    def generator(self, shape, y=None):
        with tf.variable_scope("generator") as scope:

            if self.shape_size == 32:
                s = self.shape_size
                s2, s4, s8 = int(s/2), int(s/4), int(s/8)
                print(shape.get_shape())
                # shape is (32 x 32 x 32 x 1)
                e1 = conv3d(shape, self.gf_dim, name='g_e1_conv')
                print(e1.get_shape())
                # e1 is (16 x 16 x 16 x self.gf_dim)
                e2 = self.g_bn_e2(conv3d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
                print(e2.get_shape())
                # e2 is (8 x 8 x 8 x self.gf_dim*2)
                e3 = self.g_bn_e3(conv3d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
                print(e3.get_shape())
                # e3 is (4 x 4 x 4 x self.gf_dim*4)
                e4 = self.g_bn_e4(conv3d(lrelu(e3), self.gf_dim*8, d_w=1, d_h=1, d_l=1, name='g_e4_conv'))
                print(e4.get_shape())
                # e4 is (4 x 4 x 4 x self.gf_dim*8)
                e5 = self.g_bn_e5(conv3d(lrelu(e4), self.gf_dim*16, d_w=1, d_h=1, d_l=1, name='g_e5_conv'))
                print(e5.get_shape())
                # e5 is (4 x 4 x 4 x self.gf_dim*16)
                
                # This is the decoder part
                print([self.batch_size, s8, s8, s8, self.gf_dim*8])
                self.d1, self.d1_w, self.d1_b = deconv3d(tf.nn.relu(e5), [self.batch_size, s8, s8, s8, self.gf_dim*8], d_w=1, d_h=1, d_l=1, name='g_d1', with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e4], 4)
                print(d1.get_shape())
                #d1 is (4 x 4 x 4 x self.gf_dim*8*2)

                self.d2, self.d2_w, self.d2_b = deconv3d(tf.nn.relu(d1),
                    [self.batch_size, s8, s8, s8, self.gf_dim*4], name='g_d2', d_w=1, d_h=1, d_l=1, with_w=True)
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e3], 4)
                print(d2.get_shape())
                # d2 is (4 x 4 x self.gf_dim*4*2)

                self.d3, self.d3_w, self.d3_b = deconv3d(tf.nn.relu(d2),
                    [self.batch_size, s4, s4, s4, self.gf_dim*2], name='g_d3', with_w=True)
                d3 = self.g_bn_d3(self.d3)
                d3 = tf.concat([d3, e2], 4)
                print(d3.get_shape())
                # d3 is (8 x 8 x self.gf_dim*2*2)



                self.d4, self.d4_w, self.d4_b = deconv3d(tf.nn.relu(d3),
                    [self.batch_size, s2, s2, s2, self.gf_dim*1], name='g_d4', with_w=True)
                d4 = self.g_bn_d4(self.d4)
                d4 = tf.concat([d4, e1], 4)
                print(d4.get_shape())

                self.d5, self.d5_w, self.d5_b = deconv3d(tf.nn.relu(d4),
                    [self.batch_size, s, s, s, self.output_c_dim], name='g_d5', with_w=True)
                print(self.d5.get_shape())

                # d5 is (256 x 256 x output_c_dim)
                
                # rescale self.d4
                self.d5 = tf.nn.sigmoid(self.d5)
                self.d5 = tf.multiply(3., self.d5)          # The range of sdf value is truncated into [0, 3]
                return self.d5 
    
    def discriminator(self, shape, y=None, reuse=False): 
        with tf.variable_scope("discriminator") as scope:

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            if self.shape_size == 32:
                # shape is 32 x 32 x 32 x (input_c_dim)
                h0 = lrelu(conv3d(shape, self.df_dim, name='d_h0_conv'))
                # h0 is (16 x 16 x 16 x self.df_dim)
                h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim*2, name='d_h1_conv')))
                # h1 is (8 x 8 x 8 self.df_dim*2)
                h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim*4, name='d_h2_conv')))
                # h2 is (4 x 4 x 4 x self.df_dim*4)
                h3 = lrelu(self.d_bn3(conv3d(h2, self.df_dim*8, d_w=1, d_h=1, d_l=1, name='d_h3_conv')))
                # h3 is (4 x 4 x 4 x self.df_dim*8)
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4_local = linear(tf.reshape(h3, [self.batch_size*4, -1]), 1, 'd_h3_lin_local')

                return tf.nn.sigmoid(h4), h4, h4_local


    def build_model(self, withLocalLoss=False):
        # parital shape placeholder
        self.partial_shape = tf.placeholder(tf.float32, [self.batch_size, self.shape_size, self.shape_size, self.shape_size, self.input_c_dim], name='partial_shape')
        # Complete shape placeholder
        self.target_shape = tf.placeholder(tf.float32, [self.batch_size, self.shape_size, self.shape_size, self.shape_size, self.output_c_dim], name='target_shape')
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.shape_size, self.shape_size, self.shape_size, self.output_c_dim], name='mask')


        # Generated shape 
        self.generated_shape = self.generator(self.partial_shape)
        '''
        self.real_pair = tf.concat([self.partial_shape, self.complete_shape], axis=4)
        self.fake_pair = tf.concat([self.partial_shape, self.generated_shape], axis=4)
        '''

        self.D, self.D_logits, self.D_logits_local = self.discriminator(self.target_shape, reuse=False)
        self.D_, self.D_logits_, self.D_logits_local_ = self.discriminator(self.generated_shape, reuse=True)

        # Randomly check results
        self.fake_shape = self.sampler(self.partial_shape)

        # Global histogram summary
        self.d_sum = tf.summary.histogram('d', self.D)
        self.d__sum = tf.summary.histogram('d_', self.D_) 



  
        # Global discriminator loss
        self.d_loss_real_global = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D), name='d_loss_real_global'))
        self.d_loss_fake_global = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_), name='d_loss_fake_global'))

        # Global generator loss 
        print(self.D_logits_.get_shape())
        self.g_loss_global = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_), name='g_loss_global')) 

        if withLocalLoss:
            # Local discriminator loss
            self.d_loss_real_local = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_local, labels=tf.ones_like(self.D_logits_local), name='d_loss_real_local'))
            self.d_loss_fake_local = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_local_, labels=tf.zeros_like(self.D_logits_local_), name='d_loss_fake_local'))
            # Local generator loss
            self.g_loss_local = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_local_,  labels=tf.ones_like(self.D_logits_local_), name='g_loss_local'))
            
            # SUM all the losses
            self.g_loss         =   self.g_loss_global + self.g_loss_local + self.L1_lambda * tf.reduce_mean(tf.multiply(tf.abs(self.target_shape - self.generated_shape), self.mask)) 
            self.d_loss_real    =   self.d_loss_real_global + self.d_loss_real_local 
            self.d_loss_fake    =   self.d_loss_fake_global + self.d_loss_fake_local 
            self.d_loss = self.d_loss_real + self.d_loss_fake
        else:
            print("withoutLocalLoss")
            print(self.g_loss_global.get_shape())
            print(self.target_shape.get_shape())
            print(self.generated_shape.get_shape())
            self.g_loss = self.g_loss_global + self.L1_lambda * tf.reduce_mean(tf.multiply(tf.abs(self.target_shape - self.generated_shape), self.mask)) 
            self.d_loss_real = self.d_loss_real_global 
            self.d_loss_fake = self.d_loss_fake_global 
            self.d_loss = self.d_loss_real + self.d_loss_fake 
       
       


        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, args):
        """Train shape Completion"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

        counter = 1
        start_time = time.time()

        print("checkpoint_dir = {}".format(args.checkpoint_dir))
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        # print('{:10} = {:4d}'.format('args.epoch', args.epoch))
        for epoch in xrange(args.epoch):
                      
            for idx in xrange(0, self.train_size, self.batch_size):
                batch_shape, mask = self.next_batch(self.batch_size, 'train')  # THIS IS TRAINING
                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={self.partial_shape: batch_shape[:,:,:,:,:1], self.target_shape:batch_shape[:,:,:,:,1:], self.mask: mask})
                self.writer.add_summary(summary_str, counter)
                    
                # Update G network
                _, summary_str, temp_shape = self.sess.run([g_optim, self.g_sum, self.fake_shape],
                    feed_dict={self.partial_shape: batch_shape[:,:,:,:,:1], self.target_shape:batch_shape[:,:,:,:,1:], self.mask: mask})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
               
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={self.partial_shape: batch_shape[:,:,:,:,:1], self.target_shape:batch_shape[:,:,:,:,1:], self.mask: mask})
                self.writer.add_summary(summary_str, counter)
                

                errD = self.d_loss.eval({self.partial_shape: batch_shape[:,:,:,:,:1], self.target_shape:batch_shape[:,:,:,:,1:], self.mask: mask})
                errG = self.g_loss.eval({self.partial_shape: batch_shape[:,:,:,:,:1], self.target_shape:batch_shape[:,:,:,:,1:], self.mask: mask})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, self.train_size,
                        time.time() - start_time, errD, errG))

                if np.mod(counter, 20) == 1:
                    self.sample_shapes(args.sample_dir, epoch, idx)
                    #self.sample_triplet(args.sample_dir, epoch, idx)

                if np.mod(counter, 50) == 2:
                    self.save(args.checkpoint_dir, counter)
  

    def sample_shapes(self, sample_dir, epoch, idx):
        sample_shapes, mask = self.next_batch(self.batch_size, 'random')
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_shape, self.d_loss, self.g_loss],
            feed_dict={self.partial_shape: sample_shapes[:,:,:,:,:1], self.target_shape:sample_shapes[:,:,:,:,1:], self.mask: mask}
        )
        save_shapes_asTxt(samples, './{}/train_{:02d}_{:04d}_'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
    
    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
   

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        #for i, sample_image in enumerate(sample_images):
        for i in range(100):
            sample_shapes, _ = self.next_batch(self.batch_size, 'test')  # THIS IS TRAINING
            idx = i+1
            print("sampling shape ", idx)
            samples = self.sess.run(
                self.fake_shape,
                feed_dict={self.partial_shape: sample_shapes[:,:,:,:,:1]}
            )
            save_shapes(samples, './{}/test_{:04d}_'.format(args.test_dir, idx))
            print('./{}/test_{:04d}_'.format(args.test_dir, idx))
            save_noisy_shapes(sample_shapes[:,:,:,:,:1], './{}/org_{:04d}_'.format(args.test_dir, idx))

    
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save(self, checkpoint_dir, step):
        model_name = "shapeCompletion.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def sampler(self, shape, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if self.shape_size == 32:
                s = self.shape_size
                s2, s4, s8 = int(s/2), int(s/4), int(s/8)
                print(shape.get_shape())
                # shape is (32 x 32 x 32 x 1)
                e1 = conv3d(shape, self.gf_dim, name='g_e1_conv')
                print(e1.get_shape())
                # e1 is (16 x 16 x 16 x self.gf_dim)
                e2 = self.g_bn_e2(conv3d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
                print(e2.get_shape())
                # e2 is (8 x 8 x 8 x self.gf_dim*2)
                e3 = self.g_bn_e3(conv3d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
                print(e3.get_shape())
                # e3 is (4 x 4 x 4 x self.gf_dim*4)
                e4 = self.g_bn_e4(conv3d(lrelu(e3), self.gf_dim*8, d_w=1, d_h=1, d_l=1, name='g_e4_conv'))
                print(e4.get_shape())
                # e4 is (4 x 4 x 4 x self.gf_dim*8)
                e5 = self.g_bn_e5(conv3d(lrelu(e4), self.gf_dim*16, d_w=1, d_h=1, d_l=1, name='g_e5_conv'))
                print(e5.get_shape())
                # e5 is (4 x 4 x 4 x self.gf_dim*16)
                
                # This is the decoder part
                print([self.batch_size, s8, s8, s8, self.gf_dim*8])
                self.d1, self.d1_w, self.d1_b = deconv3d(tf.nn.relu(e5), [self.batch_size, s8, s8, s8, self.gf_dim*8], d_w=1, d_h=1, d_l=1, name='g_d1', with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e4], 4)
                print(d1.get_shape())
                #d1 is (4 x 4 x 4 x self.gf_dim*8*2)

                self.d2, self.d2_w, self.d2_b = deconv3d(tf.nn.relu(d1),
                    [self.batch_size, s8, s8, s8, self.gf_dim*4], name='g_d2', d_w=1, d_h=1, d_l=1, with_w=True)
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e3], 4)
                print(d2.get_shape())
                # d2 is (4 x 4 x self.gf_dim*4*2)

                self.d3, self.d3_w, self.d3_b = deconv3d(tf.nn.relu(d2),
                    [self.batch_size, s4, s4, s4, self.gf_dim*2], name='g_d3', with_w=True)
                d3 = self.g_bn_d3(self.d3)
                d3 = tf.concat([d3, e2], 4)
                print(d3.get_shape())
                # d3 is (8 x 8 x self.gf_dim*2*2)



                self.d4, self.d4_w, self.d4_b = deconv3d(tf.nn.relu(d3),
                    [self.batch_size, s2, s2, s2, self.gf_dim*1], name='g_d4', with_w=True)
                d4 = self.g_bn_d4(self.d4)
                d4 = tf.concat([d4, e1], 4)
                print(d4.get_shape())

                self.d5, self.d5_w, self.d5_b = deconv3d(tf.nn.relu(d4),
                    [self.batch_size, s, s, s, self.output_c_dim], name='g_d5', with_w=True)
                print(self.d5.get_shape())

                # d5 is (256 x 256 x output_c_dim)
                
                # rescale self.d4
                self.d5 = tf.nn.sigmoid(self.d5)
                self.d5 = tf.multiply(3., self.d5)          # The range of sdf value is truncated into [0, 3]
                return self.d5 

