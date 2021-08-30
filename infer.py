import cv2
import torch

# Model
model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo

# Images
for f in ['zidane.jpg', 'bus.jpg']:
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
imgs = [cv2.imread('bus.jpg')[..., ::-1]]

# imgs = [img1, img2]  # batch of images

# Inference
results = model(imgs, size=1280)  # includes NMS

# Results
results.print()  
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]