import cv2
import torch
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
from PIL import Image

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed

TEXT_PROMPT = "capybara"
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2

# Replace with the path to your existing video file
video_path = "capybara.mp4"

# Replace with the desired output video file path
output_video_path = "annotated_capybara.mp4"

cap = cv2.VideoCapture(video_path)

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))
groundingdino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # preprocess image
    transformed_image = preprocess_image(frame)

    # Perform object detection on the current frame
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=transformed_image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    print(f'Found {len(boxes)} objects in the frame, boxes: {boxes}, logits: {logits}, phrases: {phrases}')
    exit(-1)
    # Annotate the frame
    # annotated_frame = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
    # annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # Write the annotated frame to the output video
    # out.write(annotated_frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()