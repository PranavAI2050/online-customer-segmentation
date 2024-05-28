
# Dictionary mapping categorical features (strings) to their vocabulary size (number of unique values) after conversion to indices.
# Used for embedding categorical features.
CATEGORICAL_FEATURES_DICT = {'VisitorType':3,'Month':10}

# List of continuous features to be bucketized (divided into ranges).
BUCKET_FEATURE_KEYS = ['Informational_Duration', 'Administrative_Duration', 'ProductRelated_Duration']

# List of continuous features treated as numerical (no bucketing).
NUMERIC_FEATURE_KEYS = ['BounceRates','ExitRates','PageValues','SpecialDay']

# List of features to be one-hot encoded (converted to binary vectors representing presence/absence in each category).
ONE_HOT_FEATURES = ['Administrative','Informational','ProductRelated','OperatingSystems','Browser','Region','TrafficType']

# Name of the boolean feature.
BOOL_FEATURES = 'Weekend'

# Dictionary mapping continuous features to the number of buckets used for bucketing.
BUCKET_FEATURE_DICT = {'Informational_Duration': 4,'Administrative_Duration':4,'ProductRelated_Duration':3}


# Number of buckets used for out-of-vocabulary (OOV) values in categorical features.
NUM_OOV_BUCKETS = 1

# Total vocabulary size (including OOV bucket).
VOCAB_SIZE = 10

# Size of the OOV bucket.
OOV_SIZE = 1


# Name of the target feature (the one the model will predict).
LABEL_KEY = 'Revenue'

def transformed_name(key):
    """
  Returns a new feature name by appending "_xf" to the original name.

  Args:
      key: The original feature name (string).

  Returns:
      A new string representing the transformed feature name.
      """
    return key + '_xf'
