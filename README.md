# Depth-Estimation-Monocular-Camera

## Real-time Depth Estimation

This repository hosts an implementation of real time depth analysis, which uses vision transformers to estimate depth from images. The application captures real-time video feed, processes it through the model, and visualizes depth maps.

## Project Overview

This project demonstrates how to estimate real-time depth estimation from pre-trained Depth Anything V2. Below are example transformations from the model on image:

| Original Image | Depth Map |
|:--------------:|:---------:|
| ![Original Image](image1.jpg) | ![Depth Map](output_depth_map_rgb.png) |

The file test2_camera_video_analysis.py estimates the real time depth on video-feed through camera, which can be integrated with any monecular camera to estimate the real-time depth estimation.


## Acknowledgement 
[Depth Anything V2](https://arxiv.org/abs/2406.09414)
