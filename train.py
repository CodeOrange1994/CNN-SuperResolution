from image_generate import *
import tensorflow as tf
from PIL import Image
import time
import pickle
import os

l2_lambda=1e-2

def get_filter_var(height,width,in_chan, out_chan,layer_name,var_dict):
    shape=[height,width,in_chan, out_chan]
    weights=tf.get_variable(name=layer_name+"_w",dtype=tf.float32,shape=shape)
    var_dict[layer_name+"_w"]=weights
    #L2 regulariztion
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(l2_lambda)(weights))

    bias=tf.get_variable(name=layer_name+"_b",dtype=tf.float32,shape=[out_chan],initializer=tf.constant_initializer(0.0))
    var_dict[layer_name+"_b"]=bias
    return weights,bias

def conv2d_layer(inputs,height,width,in_chan,out_chan,layer_name,var_dict,relu=True):
    with tf.variable_scope(layer_name):
        w,b=get_filter_var(height,width,in_chan, out_chan,layer_name,var_dict)
        conv=tf.nn.conv2d(inputs,w ,strides=[1,1,1,1],padding='VALID')
        if relu==True:
            return tf.nn.relu(tf.nn.bias_add(conv,b))
        else: return tf.nn.bias_add(conv,b)
        

def main():
    filter_dict={}
    #build the layer: 5 conv layer
    
    inputs = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
    labels = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
    test = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
    
    
    con1 = conv2d_layer(inputs,3,3,1,32,"layer1",filter_dict)
    con2 = conv2d_layer(con1,3,3,32,64,"layer2",filter_dict)
    con3 = conv2d_layer(con2,3,3,64,64,"layer3",filter_dict)
    con4 = conv2d_layer(con3,3,3,64,128,"layer4",filter_dict)
    out = conv2d_layer(con4,3,3,128,1,"layer5",filter_dict,relu=False)#90*90

    square_loss = tf.reduce_sum(tf.reduce_mean(tf.square(labels - out),0))
    tf.add_to_collection('losses', square_loss)

    total_loss = tf.add_n(tf.get_collection('losses'))
    
    learning_rate=1e-4
    learning_rate_last_layer=1e-5
    
    var1 = tf.trainable_variables()[:-2]
    var2 = tf.trainable_variables()[-2:] # variables of the last layer
   
    opt1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    opt2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_last_layer)
    grads1_sq = tf.gradients(square_loss, var1)
    grads2_sq = tf.gradients(square_loss, var2)
    grads1 = tf.gradients(total_loss, var1)
    grads2 = tf.gradients(total_loss, var2)
    clipped_grads1, _ = tf.clip_by_global_norm(grads1, 10)
    clipped_grads2, _ = tf.clip_by_global_norm(grads2, 10)
    train_op1 = opt1.apply_gradients(zip(clipped_grads1, var1))
    train_op2 = opt2.apply_gradients(zip(clipped_grads2, var2))
    train_op = tf.group(train_op1, train_op2)
    
   # train_op1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss, var_list=var1)
   # train_op2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_last_layer).minimize(total_loss, var_list=var2)
   # train_op = tf.group(train_op1, train_op2) 
    #train=tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

    

    saver=tf.train.Saver()
    savedir='./sr_save1/model.ckpt'
    
    
    data_dir="/home/thuaicourse1/ltj_code/srcnn/data/train_input/"
    #tmp_dir="./data_save/"
    epoch_dir="./data/epoch.pkl"
    pics=load_image_dir(data_dir)
    size=100
    shrink_num=10
    MAX_V=255.0
    data,label=get_train_data(create_sub_images(pics),shrink_num)
    data,label=data/MAX_V,label/MAX_V

    batch_size=64
    epoch_n=100
    batch_n=data.shape[0]//batch_size

    
    save_period=5
    reload_period=5

    if not os.path.exists(epoch_dir):
        with open(epoch_dir,'wb') as f:
            epoch=0
            pickle.dump(epoch,f)
            
    
    
    with tf.Session() as sess:
        try:
            saver.restore(sess,savedir)
            print("restore data")
        except:
            tf.global_variables_initializer().run()
            print("first initialize finished")

        for epoch in range(epoch_n):
            j,myloss=0,0.0
            start_t = time.clock()
            for batch_index in range(batch_n):
                f_dict={inputs:data[batch_index*batch_size:(batch_index+1)*batch_size,:,:,0:1],
                        labels:label[batch_index*batch_size:(batch_index+1)*batch_size,:,:,0:1] }
                
                _,losses,sqloss,w1,w2,w3,w4,w5=sess.run([train_op,total_loss,square_loss,filter_dict["layer1_w"],
                                                  filter_dict["layer2_w"],filter_dict["layer3_w"],
                                                  filter_dict["layer4_w"],filter_dict["layer5_w"],
                                                            ],
                                                    feed_dict=f_dict)

                myloss+=losses

                max1,max2,max3,max4,max5=np.max(w1),np.max(w2),np.max(w3),np.max(w4),np.max(w5)
                min1,min2,min3,min4,min5=np.min(w1),np.min(w2),np.min(w3),np.min(w4),np.min(w5)
                            
                print("max weights: %.7f %.7f %.7f %.7f %.7f"%(max1,max2,max3,max4,max5))
                print("min weights:%.7f %.7f %.7f %.7f %.7f"%(min1,min2,min3,min4,min5))
                print('total losses:',losses,"sq:",sqloss,"\n")
               
                j+=1
            finish_t=time.clock()
            

            with open(epoch_dir,'rb') as f:
                ep=pickle.load(f)
            ep+=1
            with open(epoch_dir,'wb')as f:
                pickle.dump(ep,f)
            print("epoch", ep ," total losses:" , myloss," trainning time: ", (finish_t-start_t),"s\n")    

            if (epoch+1)%save_period==0:
                print("saving weights...")
                save_path=saver.save(sess,savedir)
                
                print("variables are saved!\n")
            if (epoch+1)%reload_period==0:
                print("reloading pics......")
                pics=load_image_dir(data_dir)
                data,label=get_train_data(create_sub_images(pics),shrink_num)
                data,label=data/MAX_V,label/MAX_V
                print("reloading finished!\n")
  

main()
