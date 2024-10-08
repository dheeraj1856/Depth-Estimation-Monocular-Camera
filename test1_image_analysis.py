import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# Select the device: cuda, mps, or cpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

# Load the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
model = model.to(DEVICE).eval()

# Load the input image
raw_img = cv2.imread('image1.jpg')

# Infer depth
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

# Normalize the depth map to range [0, 255]
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

# Convert the normalized depth map to uint8
depth_normalized = depth_normalized.astype('uint8')

# Apply a color map (e.g., JET, VIRIDIS, etc.) to convert the grayscale depth map to RGB
depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Save the colorized depth map as an RGB image
cv2.imwrite('output_depth_map_rgb.png', depth_colormap)

# Optionally, display the colorized depth map
cv2.imshow('Depth Map in RGB', depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
