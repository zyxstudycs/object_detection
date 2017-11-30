# this file aims at using transfer learning, to get the result in somewhat middle part of the computational
# graph, this time I use inception_resnet_v2
import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from data_preprocessing import image_preprocessing
slim = tf.contrib.slim


def get_loss(box_confidence, confidence, box_coordinate, coordinate, box_class_label, class_label):
    # first get all the loss which grids have target in it
    filtering_mask_obj = tf.reshape(box_confidence > 0.5, [-1, 18, 32])
    
    box_confidence_obj = tf.boolean_mask(box_confidence, filtering_mask_obj)
    confidence_obj = tf.boolean_mask(confidence, filtering_mask_obj)
    box_coordinate_obj = tf.boolean_mask(box_coordinate, filtering_mask_obj)
    coordinate_obj = tf.boolean_mask(coordinate, filtering_mask_obj)
    box_class_label_obj = tf.boolean_mask(box_class_label, filtering_mask_obj)
    class_label_obj = tf.boolean_mask(box_class_label, filtering_mask_obj)
    
    coor_loss = 5 * tf.reduce_sum(tf.square(tf.subtract(box_coordinate_obj, coordinate_obj)))
    conf_loss = 5 * tf.reduce_sum(tf.square(tf.subtract(box_confidence_obj, confidence_obj)))
    class_loss = tf.reduce_sum(tf.square(tf.subtract(box_class_label_obj, class_label_obj)))
    
    # get all the loss which grids have no target in it
    filtering_mask_noobj = tf.reshape(box_confidence < 0.5, [-1, 18, 32])
    
    box_confidence_noobj = tf.boolean_mask(box_confidence, filtering_mask_noobj)
    confidence_noobj = tf.boolean_mask(confidence, filtering_mask_noobj)
    
    conf_loss_noobj = 0.5 * tf.reduce_sum(tf.square(tf.subtract(box_confidence_noobj, confidence_noobj)))
    
    loss = coor_loss + conf_loss + class_loss + conf_loss_noobj
    
    return loss

def get_class_filter(class_label_obj, box_class_label_obj):
    box_classes = tf.argmax(class_label_obj, axis=-1)
    box_class_scores = tf.argmax(box_class_label_obj, axis=-1)
    class_filter = tf.equal(box_classes, box_class_scores)
    return class_filter

def get_predicted_number(box_coordinate_mask, coordinate_mask, threshold = 0.5):
    overlap_matrix = bbox_overlap_iou(box_coordinate_mask, coordinate_mask)
    great_iou = overlap_matrix > threshold
    great_num = tf.reduce_sum(tf.cast(great_iou, tf.int32))
    return great_num



def bbox_overlap_iou(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

    xI1 = tf.maximum(x11, x21)
    yI1 = tf.maximum(y11, y21)

    xI2 = tf.minimum(x12, x22)
    yI2 = tf.minimum(y12, y22)

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (bboxes1_area + bboxes2_area) - inter_area

    return tf.maximum(inter_area / union, 0)


# write a function to calculate the recall of the predict, response for accuracy
def get_accuracy(box_confidence, confidence, box_coordinate, coordinate, box_class_label, class_label):
    filtering_mask_obj = tf.reshape(confidence > 0.5, [-1, 18, 32])

    box_confidence_obj = tf.boolean_mask(box_confidence, filtering_mask_obj)
    confidence_obj = tf.boolean_mask(confidence, filtering_mask_obj)   
    box_coordinate_obj = tf.boolean_mask(box_coordinate, filtering_mask_obj)
    coordinate_obj = tf.boolean_mask(coordinate, filtering_mask_obj)
    box_class_label_obj = tf.boolean_mask(box_class_label, filtering_mask_obj)
    class_label_obj = tf.boolean_mask(box_class_label, filtering_mask_obj)
    
    filtering_mask_obj_recall = tf.reshape(box_confidence > 0.5, [-1, 18, 32])
    confidence_ojb_recall = tf.boolean_mask(box_confidence, filtering_mask_obj_recall)

    # get total number of true object
    number_precise = tf.size(confidence_obj)
    number_recall = tf.size(confidence_ojb_recall)
    
    # get true type
    true_mask = tf.reshape(box_confidence_obj > 0.5, [-1,])
    
    box_confidence_obj = tf.boolean_mask(box_confidence_obj, true_mask)
    confidence_obj = tf.boolean_mask(confidence_obj, true_mask)   
    box_coordinate_obj = tf.boolean_mask(box_coordinate_obj, true_mask)
    coordinate_obj = tf.boolean_mask(coordinate_obj, true_mask)
    box_class_label_obj = tf.boolean_mask(box_class_label_obj, true_mask)
    class_label_obj = tf.boolean_mask(class_label_obj, true_mask)

    class_filter = get_class_filter(class_label_obj, box_class_label_obj)

    box_coordinate_mask = tf.boolean_mask(box_coordinate_obj, class_filter)
    coordinate_mask = tf.boolean_mask(coordinate_obj, class_filter)

    # get total number of predicted true object
    predicted_number = get_predicted_number(box_coordinate_mask, coordinate_mask)
    recall = tf.divide(predicted_number, number_recall)
    precise = tf.divide(predicted_number, number_precise)
    
    return recall, precise


# In[14]:


# build the computational graph from inception_resnet_v2 and extract a mid point.
def build_model():
    with tf.device('/gpu:0'):
        images = tf.placeholder(tf.float32, shape=(None, 360, 640, 3))
        box_confidence = tf.placeholder(tf.float32, shape=(None, 18, 32, 1))
        box_coordinate = tf.placeholder(tf.float32, shape=(None, 18, 32, 4))
        box_class_label = tf.placeholder(tf.float32, shape=(None, 18, 32, 4))

        images_norm = tf.divide(images, 255)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(images_norm, num_classes = 1001, is_training = False)  

    #     weight_test = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'InceptionResnetV2/Repeat_1/block17_2/Branch_1/Conv2d_0c_7x1/weights')[0]

        Mixed_5b = end_points['Mixed_5b']

    # with tf.device('/gpu:0'):
        with tf.name_scope('my_fine_tune') as scope:
            conv1 = tf.layers.conv2d(Mixed_5b, filters=320, kernel_size=(3,3), strides=2, padding = 'valid',
                                     activation = tf.nn.relu, name='conv1')
        #     weight1= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1')[0]
            batch_norm1 = tf.layers.batch_normalization(conv1, axis = -1)

            conv2 = tf.layers.conv2d(inputs=batch_norm1, filters=160, kernel_size=(3,3), strides=1,
                                    padding = 'SAME', activation = tf.nn.relu, name='conv2')
            batch_norm2 = tf.layers.batch_normalization(conv2, axis = -1)

            conv3 = tf.layers.conv2d(inputs=batch_norm2, filters = 80, kernel_size=(3,3), strides=1,
                                    padding = 'SAME', activation = tf.nn.relu, name='conv3')
            batch_norm3 = tf.layers.batch_normalization(conv3, axis = -1)

            conv4 = tf.layers.conv2d(inputs=batch_norm3, filters = 20, kernel_size=(1,3), strides=1,
                                    padding = 'VALID', activation = tf.nn.relu, name='conv4')
            batch_norm4 = tf.layers.batch_normalization(conv4, axis = -1)

            conv5 = tf.layers.conv2d(inputs=batch_norm4, filters = 9, kernel_size=(1,3), strides=1,
                                    padding = 'VALID', activation = tf.nn.relu, name='conv5')
            batch_norm5 = tf.layers.batch_normalization(conv5, axis = -1)

            confidence = tf.layers.conv2d(inputs=batch_norm5, filters = 1, kernel_size=(3,3), strides=1,
                                    padding = 'VALID', activation = tf.nn.relu, name='conv_confidence')
            coordinate = tf.layers.conv2d(inputs=batch_norm5, filters = 4, kernel_size=(3,3), strides=1,
                                    padding = 'VALID', activation = tf.nn.relu, name='conv_coordinate')
            class_label = tf.layers.conv2d(inputs=batch_norm5, filters = 4, kernel_size=(3,3), strides=1,
                                    padding = 'VALID', activation = tf.nn.relu, name='conv_class_label')

            loss = get_loss(box_confidence, confidence, box_coordinate, coordinate, box_class_label, class_label)
            recall, precise = get_accuracy(box_confidence, confidence, box_coordinate, coordinate, box_class_label, class_label)

            # define the variables that I want to train
            weight1, bias1= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1')

            weight2, bias2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv2')

            weight3, bias3= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv3')

            weight4, bias4= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv4')

            weight5, bias5= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv5')

            weight_confidence, bias_confidence= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_confidence')

            weight_coordinate, bias_coordinate= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_coordinate')

            weight_class_label, bias_class_label= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv_class_label')

            variables_to_train = [weight1,weight2,weight3,weight4,weight4,weight_confidence,weight_coordinate,weight_class_label,
                                 bias1,bias2,bias3,bias4,bias5,bias_confidence,bias_coordinate,bias_class_label]

            # define the optimizer and train operation.
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss, var_list = variables_to_train)

        # define the variables that I don't need to restore
        exclude = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits','my_fine_tune/',
                  'conv1','conv2','conv3','conv4','conv5','conv_confidence','conv_coordinate','conv_class_label',
                  'batch_normalization','batch_normalization_2','batch_normalization_3','batch_normalization_4',
                  'batch_normalization_5']

        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
        saver = tf.train.Saver(variables_to_restore)
    return saver, images, box_confidence, box_class_label, box_coordinate, train_op, recall, precise, loss, confidence
