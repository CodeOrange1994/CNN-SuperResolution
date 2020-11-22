from image_generate import *
import tensorflow as tf
from PIL import Image
import pickle


def get_filter_var(height,width,in_chan, out_chan,layer_name,var_dict):
    shape=[height,width,in_chan, out_chan]
    weights=tf.get_variable(name=layer_name+"_w",dtype=tf.float32,shape=shape)
    var_dict[layer_name+"_w"]=weights
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
    di="./pic/"
    #x=load_image_dir(di)
    size=100
    filter_num=10
    shrink_num=10
    #data,label=get_train_data(create_sub_images(x),shrink_num)
    #data,label=data/255.0,label/255.0
    filter_dict={}
    #build the layer: 5 conv layer
    
    inputs=tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
    labels=tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
    test=tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
    
    
    con1=conv2d_layer(inputs,3,3,1,32,"layer1",filter_dict)
    con2=conv2d_layer(con1,3,3,32,64,"layer2",filter_dict)
    con3=conv2d_layer(con2,3,3,64,64,"layer3",filter_dict)
    con4=conv2d_layer(con3,3,3,64,128,"layer4",filter_dict)
    out=conv2d_layer(con4,3,3,128,1,"layer5",filter_dict,relu=False)#90*90

   # loss=tf.reduce_mean(tf.square(labels - out))
   # learning_rate=1e-3
   # train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    saver=tf.train.Saver()
    savedir='./sr_save1/model.ckpt'
    testdir='test2.jpg'
    number=testdir.split('.')[0][-1]
    
    with tf.Session() as sess:
        print("begin predicting...")
        try:
            saver.restore(sess,savedir)
            print("restore data")
        except:
            #tf.global_variables_initializer().run()
            print("there is no data! ")
            return 1
        test_im=read_image(testdir)
        test_array=np.asarray([test_im],dtype=test_im.dtype)/255.0
        print(test_array)
        f_dict={inputs:test_array[:,:,:,0:1]}
        output=sess.run([out], feed_dict=f_dict)[0]
        print(output.shape)
        print(type(output))
        output=output[0,:,:,0]*255.0
        output[output>255.0]=255.0
        output[output<0.0]=0.0
        print(output.shape)
        test_im[shrink_num//2:-shrink_num//2,shrink_num//2:-shrink_num//2,0]=output
        outim=Image.fromarray(output.astype('uint8'),'L')
        outcolorim=Image.fromarray(test_im.astype('uint8'),'YCbCr')
        
        with open("./data/epoch.pkl",'rb') as f:
            eq=pickle.load(f)
        outim.save("./pred/predict_"+str(eq)+"_"+number+".jpg")
        outcolorim.save("./pred/color_predict"+str(eq)+"_"+number+".jpg")
 
main()
