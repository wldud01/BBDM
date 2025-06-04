import logging
import importlib
import os

# 데이터셋 종류, runner 종류 등 등록 기능 제공공
class Register:
    def __init__(self, registry_name):
        self.dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable")
        if key is None:
            key = value.__name__
        if key in self.dict:
            logging.warning("Key %s already in registry %s." % (key, self.__name__))
        self.dict[key] = value
    
    # 객체의 이름으로 관리
    def register_with_name(self, name):
        def register(target):
            def add(key, value):
                self[key] = value
                return value

            if callable(target):
                return add(name, target)
            return lambda x: add(target, x)
        return register

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()

# config 기반 실험 관리를 위한 클래스
class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")
    
    # 아래와 같이 하면 Registers.datasets['custom_ct_translation'] 방식으로 데이터셋을 불러올 수 있음음
    datasets = Register('datasets')
    runners = Register('runners')
