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

encoder = 'vits'  # or 'vits', 'vitb', 'vitg'

# Load the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
model = model.to(DEVICE).eval()

# Initialize video capture from the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame from the video feed
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Infer depth from the current frame
    depth = model.infer_image(frame)  # HxW raw depth map in numpy

    # Normalize the depth map to range [0, 255]
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized depth map to uint8
    depth_normalized = depth_normalized.astype('uint8')

    # Apply a color map to the depth map
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Show the colorized depth map
    cv2.imshow('Depth Map in RGB', depth_colormap)

    # Show the original video feed as well, if needed
    cv2.imshow('Original Video Feed', frame)

    # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
