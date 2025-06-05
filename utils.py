import argparse
import importlib
import omegaconf.dictconfig

from Register import Registers
from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner

def dict2namespace(config):
    """
    DictConfig (OmegaConf) 또는 일반 dict를 argparse.Namespace로 변환
    재귀적으로 모든 하위 dict도 처리
    예시: key('runner'), value('BBDMRunner')
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value   # BBDMRunner
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    """
    argparse.Namespace 객체를 일반 dict로 재귀 변환
    (주로 logging이나 저장용으로 사용)
    """
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    print(" ")
    return conf_dict


def get_obj_from_str(string, reload=False):
    """
    "module.ClassName" 문자열로부터 클래스 객체 반환
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    config 딕셔너리를 기반으로 객체 인스턴스화
    config["target"]: 클래스 경로 (예: "runners.BBDMRunner")
    config["params"]: 클래스 생성자 인자
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# default: runner -> BBDMRunner
def get_runner(runner_name, config):
    """
    Register에 등록된 Runner 클래스 반환
    Args:
        runner_name (str): 등록된 Runner 이름 (예: "BBDMRunner")
        config (Namespace): 모델 설정
    Returns:
        runner 인스턴스
    """
    runner = Registers.runners[runner_name](config)
    return runner
