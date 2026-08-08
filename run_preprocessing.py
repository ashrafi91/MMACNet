import argparse

from MMACNet.utils.configuration import Config
from MMACNet.utils.mapper import ConfigMapper

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Command line arguments
parser = argparse.ArgumentParser(description="Preprocessing datasets")
parser.add_argument(
    "--config_path", type=str, action="store", help="Path to the config file"
)
args = parser.parse_args()

# Config
config = Config(path=args.config_path)

# Preprocess!
preprocessing = ConfigMapper.get_object(
    "preprocessing_pipelines", config.preprocessing.name
)(config.preprocessing.params)
preprocessing.preprocess()
