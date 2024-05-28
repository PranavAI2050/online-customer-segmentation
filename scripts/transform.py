
import tensorflow as tf
import tensorflow_transform as tft

import constants  # Assuming constants.py is in the same directory


# Import specific constants from the constants module for better readability

_NUMERIC_FEATURE_KEYS = constants.NUMERIC_FEATURE_KEYS 
_ONE_HOT_FEATURES = constants.ONE_HOT_FEATURES 
_VOCAB_SIZE = constants.VOCAB_SIZE 
_OOV_SIZE = constants.OOV_SIZE 
_BUCKET_FEATURE_KEYS = constants.BUCKET_FEATURE_KEYS
_BUCKET_FEATURE_DICT = constants.BUCKET_FEATURE_DICT 
_CATEGORICAL_FEATURES_DICT = constants.CATEGORICAL_FEATURES_DICT 
_NUM_OOV_BUCKETS = constants.NUM_OOV_BUCKETS 
_BOOL_FEATURES = constants.BOOL_FEATURES 
_LABEL_KEY = constants.LABEL_KEY 
_transformed_name = constants.transformed_name 

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}
            

    # Scale numeric features to a range between 0 and 1.
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])
            

    # One-hot encode categorical features with vocabulary and OOV handling.
    for key in _ONE_HOT_FEATURES:
        indices = tft.compute_and_apply_vocabulary(
            inputs[key], 
            top_k=_VOCAB_SIZE, 
            num_oov_buckets=_OOV_SIZE)
        one_hot = tf.one_hot(indices, _VOCAB_SIZE + _OOV_SIZE)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, _VOCAB_SIZE + _OOV_SIZE])
            

    # Bucketize continuous features according to the bucket dictionary.
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], 
            _BUCKET_FEATURE_DICT[key],
            #always_return_num_quantiles=False
        )
    
    # One-hot encode pre-defined categorical features with vocabulary and OOV.
    for key, vocab_size in _CATEGORICAL_FEATURES_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets=_NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + _NUM_OOV_BUCKETS)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + _NUM_OOV_BUCKETS])
        
    # Keep boolean features as int64.
    outputs[_transformed_name(_BOOL_FEATURES)] = inputs[_BOOL_FEATURES]
    
    # Keep target feature as int (assuming it's the label).
    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]
    
     
    return outputs
