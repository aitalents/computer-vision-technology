import yaml


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


class Config:
    def __init__(self, conf_p="configs/config.yaml"):
        self.config_path = conf_p
        try:
            self.configuration = yaml.safe_load(open(self.config_path, "r"))
            for key, val in self.configuration.items():
                setattr(self, key, DictAsMember(val))
        except Exception as e:
            print(e)
