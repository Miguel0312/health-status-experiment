from enum import Enum, auto


class ModelType(Enum):
    NN = 1
    TREE = 2
    UNDEFINED = auto()


class FailureDetectionModel:
    def __init__(self):
        self.failure_result: list[tuple[float, float]] = []
