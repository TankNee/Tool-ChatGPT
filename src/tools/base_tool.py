import torch


class BaseTool():
    def __init__(self, llm) -> None:
        self.llm = llm

    def inference():
        raise NotImplementedError
