import torch


class BaseTool():
    def __init__(self, device) -> None:
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def inference():
        raise NotImplementedError
