
# coding: utf-8

# In[1]:


import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
label_file = 'training/label.idl'
train_file = 'training/'


# In[2]:


# write a function to plot the image and boxes.
def plot_image(pic, poses):
    
    fig,ax = plt.subplots()
    img=mpimg.imread(train_file + pic0)
    ax.imshow(img)
    
    for pos in poses:
        left_up_x = pos[0]
        left_up_y = pos[1]
        width = pos[2] - pos[0]
        height = pos[3] - pos[1]
        label = pos[4]
        
        if label == 1:
            edgecolor = 'r'
            text = 'vehicle'
        if label == 2:
            edgecolor = 'b'
            text = 'pedestrian'
        if label == 3:
            edgecolor = 'g'
            text = 'cyclist'
        if label == 20:
            edgecolor = 'y'
            text = 'traffic lights'
        
        rect = Rectangle((left_up_x,left_up_y),width,height,linewidth=1,edgecolor=edgecolor,facecolor='none')
        ax.add_patch(rect)
        ax.text(left_up_x/640, 1 - left_up_y/360, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
            
    plt.show()


# In[3]:


# define a class to process data, like get image matrix, get next batch, get label format and extract some
# information from the data
class image_preprocessing(object):  
    def __init__(self, train_file, label_file):
        self.train_file = train_file
        self.label_file = label_file
        self.pic_to_poc, self.pic_to_num, self.num_to_pic, self.num_to_pos = self.extract_data(self.label_file)
        self.images = None
        self.labels = None
        self.mini_batch = 0
        
    def define_image(self, images):
        self.images = images
        
    def get_training_data(self):
        self.train_file = train_file
        images = []
        for i in range(len(self.num_to_pic)):
            pic = self.num_to_pic[i]
            img=mpimg.imread(self.train_file + pic)
            images.append(img)
        self.images = np.array(images) 
        
    # this time I only use one anchor box in each grid. This function return 3 things. 
    # thus I am assuming every grid have most 1 object.
    def get_training_label(self):
        """
        box_cofidence: [10000, 18, 32, 1]
        box_coordinate: [10000, 18, 32, 4]
        box_class_label: [10000, 18, 32, 4]
        """
        labels = []
        
        for i in range(len(self.num_to_pos)):
            dictionary = {}
            positions = self.num_to_pos[i] 
            confidence, coordinate, class_label = self.get_one_training_label(positions)
            dictionary['confidence'] = confidence
            dictionary['coordinate'] = coordinate
            dictionary['class_label'] = class_label
            labels.append(dictionary)
        
        self.labels = labels
               
    
    def next_batch(self, batch_size = 128):
        assert(type(batch_size) == int)
        if self.mini_batch >= len(self.num_to_pic):
            self.mini_batch = 0
        images = self.images[self.mini_batch: (self.mini_batch + batch_size), :, :, :]
        if(self.labels is not None):
            labels = self.labels[self.mini_batch: (self.mini_batch + batch_size)]
        else:
            labels = None
        self.mini_batch += batch_size
            
        return images, labels
        
        
    # write a function to get the pic_to_num, pic_to_pos, num_to_pic, num_to_pos
    def extract_data(self, label_file):
        pic_to_poc = {}
        pic_to_num = {}
        num_to_pic = {}
        num_to_pos = {}
        num = 0

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines[:len(lines)]:
            jline = json.loads(line)
            for pic, pos in jline.items():
                pic_to_poc[pic] = pos
                pic_to_num[pic] = num
                num_to_pic[num] = pic
                num_to_pos[num] = pos
                num += 1
        
        return pic_to_poc, pic_to_num, num_to_pic, num_to_pos
    
     
    def get_one_training_label(self, positions):
        confidence = np.zeros((18, 32, 1), dtype=np.int32)
        coordinate = np.zeros((18, 32, 4), dtype=np.float32)
        class_label = np.zeros((18, 32, 4), dtype=np.int32)
        
        for i in range(len(positions)):
            position = positions[i]
            central_x = (position[0] + position[2]) / 2
            central_y = (position[1] + position[3]) / 2
            grid_x = int(central_x / 20)
            grid_y = int(central_y / 20)
            label = position[4]
            confidence[grid_y, grid_x, 0] = 1
            coordinate[grid_y, grid_x, :] = position[:4]
            if label == 1:
                class_label[grid_y, grid_x,0] = 1
            if label == 2:
                class_label[grid_y, grid_x,1] = 1
            if label == 3:
                class_label[grid_y, grid_x,2] = 1
            if label == 20:
                class_label[grid_y, grid_x,2] = 1
                
        return confidence, coordinate, class_label


