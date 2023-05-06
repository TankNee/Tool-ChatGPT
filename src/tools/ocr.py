
from tools.base_tool import BaseTool
from utils import prompts, logger
import pytesseract
from PIL import Image


class OCR(BaseTool):
    def __init__(self, device) -> None:
        super().__init__(device)

    @prompts(name="Optical Character Recognition",
             desc="useful when you want to recognize the text in an image."
             "The input is a path string, which is the image_path of the image."
             "The output is a string, which is the recognized text. You must strictly retain all of the output, including punctuation and spacing in your response to human.")
    def inference(self, img_path):
        img = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        logger.debug(f"Recognized text is {text}")
        return text