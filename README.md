# 🛠️ I3D Tools Dataset - Codebase

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/i3dlabiisc/I3D-Tools-Dataset)

This repository contains the official **preprocessing, postprocessing, and data analysis pipelines** used in the creation and curation of the [I3D Tools Dataset](https://huggingface.co/datasets/i3dlabiisc/I3D-Tools-Dataset). It includes scripts for:

- Dataset structuring and YOLO formatting
- Segmentation map generation and harmonization
- Image caption curation and validation
- Visualization, quality checks, and class distribution analysis

---
## 📊 Dataset Statistics

- **Number of Tool Classes:** 16  
- **Total Images:** ~35,000  
- **Image Resolution:** 1024x1024  
- **Annotations per Image:**
  - YOLOv8 bounding box format
  - Pixel-level segmentation mask
  - Natural language caption

---

## 🧰 Tool Class Names

| Class No. | Tool Name       |   | Class No. | Tool Name        |
|-----------|------------------|---|-----------|-------------------|
| 1         | ball_bearing     |   | 9         | saw               |
| 2         | gear             |   | 10        | scissors          |
| 3         | hammer           |   | 11        | screw             |
| 4         | measuring_tape   |   | 12        | screwdriver       |
| 5         | nail             |   | 13        | spring            |
| 6         | nut              |   | 14        | utility_knife     |
| 7         | oring            |   | 15        | washer            |
| 8         | plier            |   | 16        | wrench            |
---
