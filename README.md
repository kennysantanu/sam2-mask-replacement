# SAM2 Image Segmentation & Replacement

An interactive web application for image segmentation and region replacement using Meta's Segment Anything Model 2 (SAM2) and Gradio. Draw a mask on an image, let SAM2 refine it, and seamlessly blend a replacement image into the selected region.

## Features

- **Upload Images**: Load an original image and a replacement image.
- **Interactive Mask Drawing**: Paint a rough mask over the area to segment.
- **SAM2-Powered Segmentation**: Automatic mask refinement using Meta's SAM2 (with fallback to manual mask if unavailable).
- **Seamless Replacement**: The masked region is replaced and blended with the replacement image.
- **Modern Gradio UI**: Clean, user-friendly interface with real-time feedback.
- **CPU/GPU Support**: Optimized for both CPU and GPU execution.
- **Fallback Methods**: If SAM2 is unavailable, the app uses the user-drawn mask directly.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/kennysantanu/sam2-mask-replacement.git
   cd sam2-mask-replacement
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Start the app:**

   ```sh
   python app.py
   ```

2. **Open the web interface:**
   The app will launch a local Gradio server and provide a link (e.g., http://127.0.0.1:7860/).

3. **Workflow:**
   - Upload your original image.
   - Upload a replacement image (same or different size; it will be resized automatically).
   - Draw a rough mask over the area to segment.
   - Click **Process Segmentation**.
   - View the drawn mask, the refined SAM2 mask, and the final result.

## File Structure

- `app.py` — Main Gradio application.
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.11+
- [Gradio](https://gradio.app/)
- [OpenCV](https://opencv.org/)
- [Ultralytics](https://www.ultralytics.com/)

All dependencies are listed in `requirements.txt`. The SAM2 model weights will be downloaded automatically on first run.

## Development Notes

- Type hints and docstrings are used throughout the code.
- Exceptions are handled gracefully with user feedback.
- UI and business logic are cleanly separated.
- The app is optimized for both CPU and GPU environments.

## License

© 2025 Kenny Santanu. All rights reserved.

## Acknowledgements

- [Meta AI - Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- [Gradio](https://gradio.app/)

## Citation

If you use this project or SAM2 in your research, please cite:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
