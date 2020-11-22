import numpy as np
from PIL import Image
import numpy as np
import os
import random



def random_augmentation(image):
    rand_dict={0:Image.FLIP_LEFT_RIGHT,1:Image.FLIP_LEFT_RIGHT,2:PIL.Image.FLIP_LEFT_RIGHT}
    rand=random.randint(0,3)
    if rand<3:
        image=image.transpose(rand_dict[rand])
    else: pass
    return image


def read_image(image_path , as_batch=False):
    img = Image.open(image_path)
    
    if img.mode != 'YCbCr':
        img = img.convert('YCbCr')
    img_array = np.array(img)
    if as_batch: # only True when creating sub-regions
        img_array = img_array[np.newaxis, :]

    return img_array.astype('float32')

def save_image(image_array, image_path):
    img = Image.fromarray(image_array.astype('uint8'),'YCbCr')
    img.save(image_path)

    return img

def show_image(image_array):
    img = Image.fromarray(image_array.astype('uint8'),'YCbCr')
    img.show()


max_batch_img = 15

def load_image_dir(image_dir):
    print('Loading images...')
    max_batch_img = 15
    all_files = os.listdir(image_dir)
    all_img_files = [filename for filename in all_files if filename.endswith('.jpg') or filename.endswith('.png')or filename.endswith('.JPG') or filename.endswith('.bmp')]
    if len(all_img_files) > max_batch_img:
        all_img_files = random.sample(all_img_files, max_batch_img)
    imgs = []
    
    for filename in all_img_files:
        img_array = read_image(os.path.join(image_dir, filename))
        imgs.append(img_array)
    
    print('Successfully loaded %d images.' % len(imgs))
    return imgs

def create_sub_images(image_list, size=(100, 100), stride=(40, 40)):
    print('Creating sub-images...')
    sub_img_list = []

    for image in image_list:
        height = image.shape[0]
        width = image.shape[1]

        for x in range(size[0] - 1, height, stride[0]):
            for y in range(size[1] - 1, width, stride[1]):
                sub_img = image[x - size[0] + 1 : x + 1, y - size[1] + 1 : y + 1, :]
                sub_img_list.append(sub_img)
    
    sub_imgs = np.asarray(sub_img_list, dtype=image_list[0].dtype)

    print('Successfully created %d sub-images of size (%d, %d).' % (len(sub_img_list), size[0], size[1]))
    return sub_imgs


def get_label_im(image_path,shrink_num):
    im_array=read_image(image_path)
    shape=im_array.shape
    im=Image.fromarray(im_array.astype('uint8'),'YCbCr')
    new=im.resize((shape[1]-shrink_num, shape[0]-shrink_num),Image.BILINEAR)
    return np.array(new).astype("float32")

def get_train_data(sub_imgs,shrink=4,onlyY=True):
    num,w,h,c=sub_imgs.shape
    input,label=[],[]
    for i in range(num):
        im=Image.fromarray(subs[i,:,:,0].astype('uint8'),'L')
        small=im.resize((w//2,h//2),Image.BILINEAR)
        back=small.resize((w,h),Image.BILINEAR)
        input.append( np.array(back) )
        
        #la=im.resize((w-shrink,h-shrink),Image.BILINEAR)
        la_array=sub_imgs[i,shrink//2:-shrink//2,shrink//2:-shrink//2,0]
        label.append(la_array)

    input=np.asarray(input,dtype=input[0].dtype)
    label=np.asarray(label,dtype=label[0].dtype)
    
    return input[:,:,:,np.newaxis],label[:,:,:,np.newaxis]

def show_L_fromarray(array):
    if len(array.shape)==2:
        im=Image.fromarray(array.astype('uint8'),'L')
        im.show()
    elif len(array.shape) ==3:
        im=Image.fromarray(array[:,:,0].astype('uint8'),'L')
        im.show()
    else:
        print("wrong!")
