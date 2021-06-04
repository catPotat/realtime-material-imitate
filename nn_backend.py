import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
MODEL_PATH = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
MODEL_PATH = f'{os.path.dirname(os.path.realpath(__file__))}/magentaStyleTransModel'

print('Loading model...')
hub_model = hub.load(MODEL_PATH)
print('Model loaded.')


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    # print(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def load_img_from_cv(cv2_im):
    img = tf.image.convert_image_dtype(cv2_im, tf.float32)
    img = img[tf.newaxis, :]
    return img

def transfer(content, style):
    tensor = hub_model(tf.constant(content), tf.constant(style))[0]
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor



if __name__ == "__main__":
    # test
    im = load_img('')
    # print(im)
    # cv2_im = cv2.imread('')
    # im = load_img_from_cv(cv2_im)
    a = transfer(im, im)[0]
    cv2.imshow(f'magenta', a)
