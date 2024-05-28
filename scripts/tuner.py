# Define imports
from kerastuner.engine import base_tuner
import kerastuner as kt
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft
import constants

# Declare namedtuple field names
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# Label key
_LABEL_KEY = constants.LABEL_KEY 

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
_transformed_name = constants.transformed_name 

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


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
              batch_size=32) -> tf.data.Dataset:
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

  # Get feature specification based on transform output
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  # Create batches of features and labels
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=_transformed_name(_LABEL_KEY))
  
  return dataset
 
def _wide_and_deep_classifier_builder(hp):
  """Build a simple keras wide and deep model using the Functional API.

  Args:
  Builds a wide_and_deep_classifier model with hyperparameter tuning
  hp: Hyperparameter object from KerasTuner
    wide_columns: Feature columns wrapped in indicator_column for wide (linear)
      part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

  Returns:
  A compiled Keras model with tuned hyperparameters
    A Wide and Deep Keras model
  """
# Define separate input layers for numeric, bucketized, boolean, and categorical features

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

  #define number of units in 4 layers of deep network as hyperparameter
  hp_deep_layer = hp.Int('deep_layer', min_value=5, max_value=20, step=5)
  deep = tf.keras.layers.Dense(hp_deep_layer)(deep)

  # Concatenate categorical inputs
  wide = tf.keras.layers.concatenate(input_categorical)

  # Create shallow dense network for categorical inputs
  #define num units of shallow dense layer as hyperparameter
  hp_wide_units = hp.Int('wide_layer', min_value=20, max_value=90, step=10)
  wide = tf.keras.layers.Dense(hp_wide_units, activation='relu')(wide)

  # Combine wide and deep networks
  combined = tf.keras.layers.concatenate([deep, wide])
  hp_concat_units = hp.Int('concat_layer', min_value=4, max_value=20, step=4)
  combined = tf.keras.layers.Dense(hp_concat_units, activation='relu')(combined)
                                              
  # Define output of binary classifier
  output = tf.keras.layers.Dense(
      1, activation='sigmoid')(combined)

  # Setup combined input
  input_layers = input_numeric + input_categorical

  # Create the Keras model
  model = tf.keras.Model(input_layers, output)

  #define learning rate as hyperparameter
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  # Define training parameters
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      metrics=['binary_accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
  
  # Print model summary
  model.summary()

  return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.

      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """

  # Define tuner search strategy
  tuner = kt.Hyperband(_wide_and_deep_classifier_builder,
                     objective=kt.Objective("val_binary_accuracy", direction="max"),
                     max_epochs=10,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')

  # Load transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  # Use _input_fn() to extract input features and labels from the train and val set
  train_set = _input_fn(fn_args.train_files[0], tf_transform_output,10)
  val_set = _input_fn(fn_args.eval_files[0], tf_transform_output,10)


  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps,
          #'class_weight': {0: 0.6, 1: 3.2}
      }
  )
