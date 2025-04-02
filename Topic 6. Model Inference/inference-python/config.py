from pydantic import BaseSettings, Field


class AppConfig(BaseSettings):

    # models
    DETECTOR_MODEL_PATH: str = Field(default='/path', env='DETECTOR_MODEL_PATH')
    CLASSIFIER_MODEL_PATH: str = Field(default='/path', env='CLASSIFIER_MODEL_PATH')

    # app config
    PORT: int = Field(default=5001, env='PORT')
    APP_NAME: str = Field(default='cv-api', env='APP_NAME')
    DEBUG: bool = Field(default='True', env='DEBUG')
