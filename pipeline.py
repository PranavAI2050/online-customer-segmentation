import os
from typing import List
from absl import logging

import tensorflow as tf
from tfx import v1 as tfx
from google.protobuf import text_format
import tensorflow_model_analysis as tfma
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2

import tfx_addons as tfxa
from tfx_addons.sampling.component import Sampler

#define pipeline name
_pipeline_name = 'online_shoppers_intention'

#define _data_root for data ingestion
_data_root = 'data'

#define path to custom modules
module_root = 'scripts'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.

#Transform module
_transform_module_file = os.path.join(module_root, 'transform.py')
_tuner_module_file = os.path.join(module_root, 'tuner.py')
_trainer_module_file = os.path.join(module_root, 'trainer.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = 'serving_model'

#pipeline root to store all pipeline related artifacts
_pipeline_root = 'pipeline/'

#define metadastore location 
_metadata_path = os.path.join(_pipeline_root,_pipeline_name,
                              'metadata.db')
enable_cache = False
# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


#define function to create pipeline 
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     enable_cache: bool,
                     module_transform_file: str, module_trainer_file : str,module_tuner_file: str ,
                     serving_model_dir: str,
                     metadata_path: str,
                     beam_pipeline_args: List[str]) -> tfx.dsl.Pipeline:
  """Implements the online_shoppers_intention pipeline with TFX."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  output = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                 example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
             ]))
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics'])

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
    
  #Undersampling the data using tfx addons
  sampled_example_gen = Sampler(input_data=example_gen.outputs['examples'],
                   sampling_strategy=tfxa.sampling.SamplingStrategy.OVERSAMPLE,
                   splits=['train'],
                   label='Revenue')

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(
      examples=sampled_example_gen.outputs['output_data'],
      schema=schema_gen.outputs['schema'],
      module_file=os.path.abspath(module_transform_file))
    
  #Performs hyperparameter tuning on the wide and deep network

  tuner = tfx.components.Tuner(
    module_file=module_tuner_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    #specifing num steps as num of epochs is not defined in tuner module
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
    )

  # Uses user-provided Python function that implements a model.
  trainer = tfx.components.Trainer(
    module_file=os.path.abspath(module_trainer_file),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    schema=schema_gen.outputs['schema'],
    train_args=tfx.proto.TrainArgs(splits=['train']),
    eval_args=tfx.proto.EvalArgs(splits=['eval']))


  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  
  eval_config = text_format.Parse("""
  ## Model information
  model_specs {
    # This assumes a serving model with signature 'serving_default'.
    signature_name: "serving_default",
    label_key: "Revenue"
  }

  ## Post training metric information
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics {
      class_name: "BinaryAccuracy"
      threshold {
        # Ensure that metric is always > 0.5
        value_threshold {
          lower_bound { value: 0.5 }
        }
        # Ensure that metric does not drop by more than a small epsilon
        # e.g. (candidate - baseline) > -1e-10 or candidate > baseline - 1e-10
        change_threshold {
          direction: HIGHER_IS_BETTER
          absolute { value: -1e-10 }
        }
      }
    }
    metrics {
      class_name: "AUC"
      threshold {
        # Ensure that metric is always > 0.5
        value_threshold {
          lower_bound { value: 0.5 }
        }
        # Ensure that metric does not drop by more than a small epsilon
        # e.g. (candidate - baseline) > -1e-10 or candidate > baseline - 1e-10
        change_threshold {
          direction: HIGHER_IS_BETTER
          absolute { value: -1e-10 }
        }
      }
    }
    metrics {
      class_name: "Recall"
      threshold {
        # Ensure that metric is always > 0.5
        value_threshold {
          lower_bound { value: 0.5 }
        }
        # Ensure that metric does not drop by more than a small epsilon
        # e.g. (candidate - baseline) > -1e-10 or candidate > baseline - 1e-10
        change_threshold {
          direction: HIGHER_IS_BETTER
          absolute { value: -1e-10 }
        }
      }
    }
    
  }

  ## Slicing information
  slicing_specs {}  # overall slice
  slicing_specs {
    feature_keys: ["Region"]
  }
  slicing_specs {
    feature_keys: ["VisitorType"]
  }
  slicing_specs {
    feature_keys: ["Weekend"]
  }
""", tfma.EvalConfig())
  #evaluator to evaluate the model ,based on baseline model  and eval config 
  evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, example_validator, sampled_example_gen,
          transform,tuner,
          trainer, model_resolver, evaluator, pusher
      ],
      enable_cache=enable_cache,
      metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)

#define function to run the pipeline locally
def run_pipeline():
    pipeline = _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        enable_cache=enable_cache,
        module_transform_file=_transform_module_file,
        module_trainer_file=_trainer_module_file,
        module_tuner_file=_tuner_module_file,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
        beam_pipeline_args=None
    )
    tfx.orchestration.LocalDagRunner().run(pipeline)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()
