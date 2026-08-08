import copy
import os
import re

import yaml

from MMACNet.utils.mapper import ConfigMapper

# Matches ${VAR} or ${VAR:-default}, e.g. so config files can read
# "${MIMIC_CSV_DIR:-datasets/mimic3/csv}" instead of hardcoding a path
# that's only valid on one machine.
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}")


def _expand_env_vars(value):
    """Recursively expand ${VAR} / ${VAR:-default} in strings loaded from YAML."""
    if isinstance(value, str):
        def _replace(match):
            var_name, _, default = match.groups()
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_yaml(path):
    """
    Function to load a yaml file and
    return the collected dict(s)

    Parameters
    ----------
    path : str
        The path to the yaml config file

    Returns
    -------
    result : dict
        The dictionary from the config file. String values support shell-like
        environment variable expansion, e.g. "${MIMIC_CSV_DIR:-datasets/mimic3/csv}"
        resolves to the MIMIC_CSV_DIR env var if set, otherwise the default after ":-".
    """

    assert isinstance(path, str), "Provided path is not a string"
    try:
        f = open(path, "r")
        result = yaml.load(f, Loader=yaml.Loader)
    except FileNotFoundError as e:
        # Adding this for future functionality
        raise e
    return _expand_env_vars(result)


def convert_params_to_dict(params):
    dic = {}
    for k, v in params.as_dict():
        try:
            obj = ConfigMapper.get_object("params", v)
            dic[k] = v
        except:
            print(
                f"Undefined {v} for the given key: {k} in mapper,"
                " storing original value"
            )
            dic[k] = v
        return dic


class Config:
    """Config Class to be used with YAML configuration files

    This class can be used to address keys as attributes.
    Ensure that there are no spaces between the keys.
    Only objects of type dict can be converted to config.

    Attributes
    ----------
    _config : dict,
        The dictionary which is formed from the
        yaml file or custom dictionary

    Methods
    -------
    as_dict(),
        Return the config object as dictionary

        Possible update:
        ## Can be converted using __getattr__ to use **kwargs
        ## with the Config object directly.

    set_value(attr,value)
        Set the value of a particular attribute.
    """

    def __init__(self, *, path=None, dic=None):
        """
        Initializer for the Config class

        Needs either path or the dict object to create the config

        Parameters
        ----------
        path: str, optional
            The path to the config YAML file.
            Default value is None.
        dic : dict, optional
            The dictionary containing the configuration.
            Default value is None.
        """
        if path:
            self._config = load_yaml(path)
        elif dic:
            self._config = dic
        else:
            raise Exception(
                "Need either path or dict object to instantiate object."
            )
        # self.keys = self._config.keys()

    def __getattr__(self, attr):
        """
        Get method for Config class. Helps get keys as attributes.

        Parameters
        ----------
        attr: The attribute name passed as <object>.attr

        Returns
        -------
        self._config[attr]: object or Config object. The value of the given key
                            if it exists.
                            If the value is a dict object, a Config object of
                            that dict is returned. Otherwise, the exact value is
                            returned.

        Raises
        ------

        AttributeError() if the given key is not defined.
        """
        if attr in super().__getattribute__('_config'):
            if isinstance(super().__getattribute__('_config')[attr], dict):
                return Config(dic=super().__getattribute__('_config')[attr])
            elif isinstance(super().__getattribute__('_config')[attr], list):
                return [
                    Config(dic=e) if isinstance(e, dict) else e
                    for e in self._config[attr]
                ]
            else:
                return super().__getattribute__('_config')[attr]
        else:
            raise AttributeError(f"Key:{attr} not defined.")

    def set_value(self, attr, value):
        """
        Set method for Config class. Helps set keys in the _config.

        Parameters
        ----------
        attr: The attribute name passed as <object>.attr
        value: The value to be stored as the attr.
        """
        self._config[attr] = value

    def __str__(self):
        """Function to print the dictionary
        contained in the object."""
        return self._config.__str__()

    def __repr__(self):
        return f"Config(dic={self._config})"

    def __deepcopy__(self, memo):
        return Config(dic=copy.deepcopy(self._config))

    def as_dict(self):
        """Function to get the config as dictionary object"""
        return dict(self._config)
