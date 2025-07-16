import gradio as gr
import numpy as np
import cv2
from PIL import Image
from ultralytics import SAM

class ImageSegmentationApp:
    def __init__(self) -> None:
        """Initialize the segmentation app and load the SAM2 model with fallback."""
        try:
            # Attempt to load the SAM2 model weights
            self.model = SAM("sam2.1_t.pt")
            self.model_available = True  # Model loaded successfully
        except Exception as e:
            # If loading fails, set model as unavailable and print error
            print(f"Failed to load SAM2 model: {e}")
            self.model = None
            self.model_available = False

    def process_segmentation(
        self, 
        image_editor: dict, 
        replacement_image: Image.Image
    ) -> list[Image.Image | None] | None:
        """
        Process the segmentation and replacement using the drawn mask and SAM2 model.
        Returns [drawn_mask, sam_mask, result_image, markdown_message].
        """
        # Check if both images are provided
        if image_editor["background"] is None or replacement_image is None:
            return [None, None, None, "**❌ Error:** Please upload both images."]
        try:
            # Extract the original image and the user-drawn mask
            original_image = image_editor["background"]
            drawn_mask = image_editor["layers"][0]
            
            # Use the alpha channel of the mask as the binary mask
            drawn_mask = drawn_mask.split()[-1]
            drawn_mask_np = np.array(drawn_mask)

            # Find contours in the mask to determine segmentation points
            points = []
            contours, _ = cv2.findContours(drawn_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    # Use centroid of contour as a point
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    points.append([cx, cy])
                else:
                    # Fallback: use the first point if the area is zero
                    x, y = contour[0][0]
                    points.append([float(x), float(y)])

            # If no points are found, return original image and a message indicating no mask was drawn         
            if not points:
                return [None, None, original_image, "**❌ Error:** No mask drawn. Please draw a mask on the original image."]

            # If the SAM2 model is unavailable, use the drawn mask directly
            if not self.model_available or not self.model:
                sam_mask = drawn_mask
                model_message = "**⚠️ Warning:** SAM2 model unavailable, using drawn mask as mask."
            else:
                # Run the SAM2 model to refine the mask
                results = self.model(
                    source=original_image,
                    points=[points],
                )
                # Extract the mask from the model output
                result_numpy_arr = results[0].masks.data.numpy()
                sam_mask_arr = np.squeeze(result_numpy_arr)
                sam_mask_arr = (sam_mask_arr * 255).astype(np.uint8)  # Convert bool to uint8
                sam_mask = Image.fromarray(sam_mask_arr)
                model_message = "**✅ Success:** Segmentation completed with SAM2."

            # Resize the replacement image to match the original image size
            replacement_image = replacement_image.resize(original_image.size)
            # Composite the replacement image onto the original using the mask
            result_image = Image.composite(replacement_image, original_image, sam_mask)

            return [drawn_mask, sam_mask, result_image, model_message]
        except Exception as e:
            # Catch and report any errors during segmentation
            print(f"Segmentation error: {e}")
            return [None, None, None, f"**❌ Error:** Segmentation error: {e}"]

    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface"""
        with gr.Blocks(title="SAM2 Image Segmentation & Replacement", theme=gr.themes.Soft(), css=".center-status-message {text-align: center;}") as demo:
            # App title and instructions
            gr.Markdown(
                f"""
                # 🎨 SAM2 Image Segmentation & Replacement
                
                Upload an original image and a replacement image, then draw a rough mask on the original image.
                
                **Instructions:**
                1. Upload your original image
                2. Upload your replacement image  
                3. Draw a mask on the original image by painting over the area you want to replace
                4. Click "Process Segmentation" to see the result
                """
            )
            gr.Markdown("### 📸 Upload Images")
            with gr.Row():
                with gr.Column():
                    # ImageMask for original image and mask drawing
                    image_editor = gr.ImageMask(
                        label="Original Image",
                        type="pil",
                        height=400
                    )
                with gr.Column():
                    # Upload for replacement image
                    replacement_image = gr.Image(
                        label="Replacement Image",
                        type="pil",
                        height=400
                    )
            with gr.Row():
                # Button to trigger segmentation
                process_btn = gr.Button("🚀 Process Segmentation", variant="primary", size="lg")
            with gr.Row():
                # Status message for feedback
                status_message = gr.Markdown(value="", elem_id="status_message", elem_classes=["center-status-message"])
            with gr.Row():
                # Display the drawn mask, SAM2 mask, and result image
                drawn_mask = gr.Image(
                    label="Drawn Mask", 
                    type="pil",
                    height=400
                )
                result_mask = gr.Image(
                    label="SAM2 Mask",
                    type="pil",
                    height=400
                )
                result_image = gr.Image(
                    label="Result", 
                    type="pil",
                    height=400
                )
            with gr.Row():
                # Display copywrite information
                gr.Markdown(
                    value="© 2025 Kenny Santanu. All rights reserved.",
                    elem_classes=["center-status-message"]
                )

            # Connect button click to segmentation function
            process_btn.click(
                fn=self.process_segmentation,
                inputs=[image_editor, replacement_image],
                outputs=[drawn_mask, result_mask, result_image, status_message]
            )
        return demo

def main() -> None:
    """Main function to run the application"""
    # Instantiate the app
    app = ImageSegmentationApp()
    # Create the Gradio interface
    demo = app.create_interface()
    # Launch the interface (web server)
    demo.launch(
        show_api=False
    )

# Run the app if this script is executed directly
if __name__ == "__main__":
    main()
