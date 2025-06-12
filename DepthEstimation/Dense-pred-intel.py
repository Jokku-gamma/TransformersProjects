from transformers import DPTImageProcessor,DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv()

HUGGING_FACE_TOKEN=os.environ.get("HUGGING_FACE_TOKEN")
processor=DPTImageProcessor.from_pretrained("Intel/dept-large",token=HUGGING_FACE_TOKEN)
model=DPTForDepthEstimation.from_pretrained("Intel/dpt-large",token=HUGGING_FACE_TOKEN)
path="road.jpeg"
image=Image.open(path)
inputs=processor(images=image,return_tensors='pt')

with torch.no_grad():
    outputs=model(**inputs)
    pred_depths=outputs.predicted_depth

pred=torch.nn.functional.interpolate(
    pred_depths.unsqueeze(1),
    size=image.size[::-1],
    mode='bicubic',
    align_corners=False,
)

output=pred.squeeze().cpu().numpy()
formatted=(output*255/np.max(output)).astype("uint8")
depth_map=Image.fromarray(formatted)

plt.imshow(depth_map,cmap='inferno')
plt.colorbar()
plt.show()
