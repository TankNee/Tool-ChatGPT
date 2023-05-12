from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from tools.base_tool import BaseTool
from utils import get_output_path, prompts, logger
import os
import torch


class TableStructureRecognition(BaseTool):

    def __init__(self, model_name, cache_dir, device, llm) -> None:
        super().__init__(llm)
        self.device = device
        model_path = snapshot_download(repo_id=model_name,
                                       cache_dir=cache_dir,
                                       local_files_only=True)
        self.model = TableTransformerForObjectDetection.from_pretrained(
            model_path).to(self.device)
        logger.debug(f"Model loaded from {model_path} to {self.device}")
        self.model.eval()

    def _plot_bboxes(self, img: Image, results: dict):
        draw = ImageDraw.Draw(img)
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]
        for score, label, (xmin, ymin, xmax,
                           ymax) in zip(scores.tolist(), labels.tolist(),
                                        boxes.tolist()):
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red")
            text = self.model.config.id2label[label]
            draw.text((xmin, ymin), f"{text} {score:.2f}")

        return img

    @prompts(
        name="Table Structure Recognition",
        desc="useful when you want to recognize the structure of a table."
        "The input is a string, which is the table cropped image path."
        "The output is a string, which is the recognized table image path. You must strictly retain all of the output, including punctuation and spacing in your response to human."
    )
    def inference(self, doc_path: str):
        if not os.path.isfile(doc_path):
            logger.error(f"Document {doc_path} is not a file.")
            return "Tool's input is not a valid file path, please reconsider your input."

        feature_extractor = DetrFeatureExtractor()
        doc_img = Image.open(doc_path).convert("RGB")
        encodings = feature_extractor(doc_img,
                                      return_tensors="pt").to(self.device)
        logger.debug(
            f"Feature encodings shape is {encodings['pixel_values'].shape}")
        with torch.no_grad():
            outputs = self.model(**encodings)

        target_sizes = [doc_img.size[::-1]]
        results = feature_extractor.post_process_object_detection(
            outputs, threshold=0.6, target_sizes=target_sizes)
        # plot bounding boxes
        out_img = self._plot_bboxes(doc_img, results[0])
        # save to local
        out_path = get_output_path(doc_path)
        out_img.save(out_path)
        logger.debug(f"Result saved to {out_path}")

        return out_path