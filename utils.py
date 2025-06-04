import argparse
import importlib
import omegaconf.dictconfig

from Register import Registers
from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner

# config 파일 내용 dict로 변환, 예시: key('runner'), value('BBDMRunner') 형태로 변환
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value   # BBDMRunner
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    print(" ")
    return conf_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# default: runner -> BBDMRunner
def get_runner(runner_name, config):
    runner = Registers.runners[runner_name](config)
    return runner
