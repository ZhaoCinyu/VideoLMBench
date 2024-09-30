# GroudingDINO on video frames

## setup environments

- Install ./GroundingDINO
- Download model weights
```bash
mkdir GroundingDINO/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
mv groundingdino_swint_ogc.pth GroundingDINO/weights/
```
- Use dino_video_frame_grounding.py to generate video frames grouding labels