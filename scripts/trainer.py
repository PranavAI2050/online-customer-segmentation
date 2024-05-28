
from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.public.tfxio import TensorFlowDatasetOptions

# import same constants from transform module
import constants

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


def _gzip_reader_fn(filenames):
  '''Load compressed dataset
  
  Args:
    filenames - filenames of TFRecords to load

  Returns:
    TFRecordDataset loaded from the filenames
  '''

  # Load the dataset. Specify the compression type since it is saved as `.gz`
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=150) -> tf.data.Dataset:
  '''Create batches of features and labels from TF Records

  Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output - transform output graph
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    batch_size - An int representing the number of records to combine in a single batch.

  Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
  '''

  # Get post-transfrom feature spec
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  # Create batches of data
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=_transformed_name(_LABEL_KEY)
      )
  
  return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  # Get transformation graph
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    # Get pre-transform feature spec
    feature_spec = tf_transform_output.raw_feature_spec()

    # Pop label since serving inputs do not include the label
    feature_spec.pop(_LABEL_KEY)

    # Parse raw examples into a dictionary of tensors matching the feature spec
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    # Transform the raw examples using the transform graph
    transformed_features = model.tft_layer(parsed_features)

    # Get predictions using the transformed features
    return model(transformed_features)

  return serve_tf_examples_fn
 


def _wide_and_deep_classifier(hp):
  """Build a simple keras wide and deep model using the Functional API.

  Args:
    wide_columns: Feature columns wrapped in indicator_column for wide (linear)
      part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

  Returns:
    A Wide and Deep Keras model
  """

  # Define input layers for numeric keys
  input_numeric = [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(1,), dtype=tf.float32)
      for colname in _NUMERIC_FEATURE_KEYS
  ]

  #define input layers for bucketize features
  input_numeric += [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(1,), dtype=tf.float32)
      for colname in _BUCKET_FEATURE_KEYS
  ]

  #adding boolean features to inputs
  input_numeric += [tf.keras.layers.Input(name=_transformed_name(_BOOL_FEATURES), shape=(1,), dtype=tf.float32)]
    
  # Define input layers for vocab keys
  input_categorical = [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(_VOCAB_SIZE + _OOV_SIZE,), dtype=tf.float32)
      for colname in _ONE_HOT_FEATURES
  ]

  # Define input layers for bucket key
  input_categorical += [
      tf.keras.layers.Input(name=_transformed_name(colname), shape=(num_buckets + _NUM_OOV_BUCKETS,), dtype=tf.float32)
      for colname, num_buckets in _CATEGORICAL_FEATURES_DICT.items()
  ]

  # Concatenate numeric inputs
  deep = tf.keras.layers.concatenate(input_numeric)

  # Create deep dense network for numeric inputs 
  hp_deep_layer = hp.get('deep_layer') 
  deep = tf.keras.layers.Dense(hp_deep_layer)(deep)

  # Concatenate categorical inputs
  wide = tf.keras.layers.concatenate(input_categorical)

  # Create shallow dense network for categorical inputs
  hp_wide_units = hp.get('wide_layer')  
  wide = tf.keras.layers.Dense(hp_wide_units, activation='relu')(wide)

  # Combine wide and deep networks
  combined = tf.keras.layers.concatenate([deep, wide])
  hp_concat_units = hp.get('concat_layer')
  combined = tf.keras.layers.Dense(hp_concat_units, activation='relu')(combined)
                                              
  # Define output of binary classifier
  output = tf.keras.layers.Dense(
      1, activation='sigmoid')(combined)

  # Setup combined input
  input_layers = input_numeric + input_categorical

  # Create the Keras model
  model = tf.keras.Model(input_layers, output)
    
  hp_learning_rate = hp.get('learning_rate')

  # Define training parameters
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      metrics=['binary_accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
  
  # Print model summary
  model.summary()

  return model

# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Defines and trains the model.
  
  Args:
    fn_args: Holds args as name/value pairs. Refer here for the complete attributes: 
    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
  """

  # Get transform output (i.e. transform graph) wrapper
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Create batches of train and eval sets
  train_dataset = _input_fn(fn_args.train_files[0], tf_transform_output, 12)
  eval_dataset = _input_fn(fn_args.eval_files[0], tf_transform_output, 12)
    
  hp = fn_args.hyperparameters.get('values')
  # Build the model
  model = _wide_and_deep_classifier(
      # Construct layers with tuned parameters
      hp
  )
  
  # Callback for TensorBoard
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')


  # Train the model
  model.fit(
      train_dataset,
      #class_weight={0: 0.6, 1: 3.2},
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])
  

  # Define default serving signature
  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  

  # Save model with signature
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
