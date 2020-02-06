import tensorflow as tf

def parse_function(example_proto):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'image_shape': tf.FixedLenFeature([], tf.string),
    }

    content = tf.parse_single_example(example_proto, features=features)

    content['image_shape'] = tf.decode_raw(content['image_shape'], tf.int32)
    content['image'] = tf.decode_raw(content['image'], tf.int16)
    content['image'] = tf.reshape(content['image'], content['image_shape'])

    return content['image']
