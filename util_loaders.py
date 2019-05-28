import os
import tensorflow as tf



def clean_models_log_files(path_to_model_dir):
    """
    Every time the model architecture changes we must clean up
    in order to prevent loading wrong shapes
    """
    os.system("rm -r " + path_to_model_dir)
    
    
    
def parser(record):
    
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[299, 299])
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label


def input_fn(filenames):
  
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
    
    dataset = dataset.apply(
      #tf.contrib.data.shuffle_and_repeat(1024, 1)
      tf.data.experimental.shuffle_and_repeat(1024,1)
  )
    dataset = dataset.apply(
      #tf.contrib.data.map_and_batch(parser, 32)
      tf.data.experimental.map_and_batch(parser,32)
  )
    #dataset = dataset.map(parser, num_parallel_calls=12)
    #dataset = dataset.batch(batch_size=1000)
    dataset = dataset.prefetch(buffer_size=2)
    
    return dataset

