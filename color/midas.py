import cv2
import torch
import numpy as np

class Midas():
    def __init__(
        self,
        model_type: str,
    ):
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(self.device)
        self.model = model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def get_depth_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        return output