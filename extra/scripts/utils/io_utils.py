import os

import yaml
from sklearn.utils import Bunch


def parse_yaml_file(yaml_file: str) -> Bunch:

    yaml_file = os.path.expanduser(yaml_file)

    # NOTE: only the first layer of keys in yaml can be visited by value-mode
    d = yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader)
    return Bunch(**d)
