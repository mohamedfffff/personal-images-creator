import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from bidi.algorithm import get_display
import arabic_reshaper
import onnxruntime as ort
import os
from io import BytesIO

# Constants for image sheet creation
A5_WIDTH = 1748
A5_HEIGHT = 2480
BACKGROUND_COLOR = (255, 255, 255)  # white
TEXT_COLOR = (0, 0, 0)  # black

def remove_background(image_input, input_size=(320, 320), return_mask=False, top_margin_percent=0.1, dpi=300):
    """
    Removes the background from an image using a combination of U2Net and MODNet AI models.

    Args:
        image_input (str, bytes, bytearray, BytesIO): Path to the image file, or image data as bytes.
        input_size (tuple): Size to which the image is resized for the AI models.
        return_mask (bool): If True, returns only the foreground mask as bytes.
        top_margin_percent (float): Percentage of the original image height to add as top margin.
        dpi (int): Dots per inch for the output image (currently not directly used in processing).

    Returns:
        bytes: The image with the background removed (as PNG bytes) or the foreground mask (as PNG bytes).

    Raises:
        FileNotFoundError: If the AI model files are not found at the specified paths.
        ValueError: If the input type is unsupported or if image loading fails.
        Exception: For other errors during processing.
    """
    try:
        # Fixed paths to the models
        u2net_path = r"C:\SnapSheet\models\u2net.onnx"
        modnet_path = r"C:\SnapSheet\models\modnet_photographic_portrait_matting.onnx"

        # Check if the model files exist
        if not os.path.isfile(u2net_path):
            raise FileNotFoundError(f"Model file not found: {u2net_path}")

        if not os.path.isfile(modnet_path):
            raise FileNotFoundError(f"Model file not found: {modnet_path}")

        # Create ONNX inference sessions
        u2net_session = ort.InferenceSession(u2net_path)
        modnet_session = ort.InferenceSession(modnet_path)

        # Read the image
        if isinstance(image_input, str):
            image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        elif isinstance(image_input, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_UNCHANGED)
        elif isinstance(image_input, BytesIO):
            image = cv2.imdecode(np.frombuffer(image_input.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("Unsupported input type. Use path, bytes, or BytesIO.")

        if image is None:
            raise ValueError("Failed to load image.")

        # Save the original image (BGR)
        original_image_bgr = image.copy()
        if original_image_bgr.ndim == 2:
            original_image_bgr = cv2.cvtColor(original_image_bgr, cv2.COLOR_GRAY2BGR)
        elif original_image_bgr.shape[2] == 4:
            original_image_bgr = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGRA2BGR)

        # Convert the image to RGB for processing
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image.shape[:2]

        # ===== Processing with U2Net =====
        resized = cv2.resize(image, input_size)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Run U2Net
        u2net_output = u2net_session.run(
            [u2net_session.get_outputs()[0].name],
            {u2net_session.get_inputs()[0].name: input_tensor}
        )[0]

        mask = u2net_output.squeeze()
        mask = cv2.resize(mask, (original_w, original_h))
        mask = (mask * 255).astype(np.uint8)

        # ===== Processing with MODNet =====
        # Prepare image for MODNet (512x512)
        modnet_size = (512, 512)
        resized_modnet = cv2.resize(image, modnet_size)
        resized_modnet = resized_modnet.astype(np.float32) / 255.0
        input_tensor_modnet = np.transpose(resized_modnet, (2, 0, 1))
        input_tensor_modnet = np.expand_dims(input_tensor_modnet, axis=0)

        # Run MODNet
        modnet_output = modnet_session.run(
            [modnet_session.get_outputs()[0].name],
            {modnet_session.get_inputs()[0].name: input_tensor_modnet}
        )[0]

        modnet_mask = modnet_output.squeeze()
        modnet_mask = cv2.resize(modnet_mask, (original_w, original_h))
        modnet_mask = (modnet_mask * 255).astype(np.uint8)

        # ===== Combining results for better quality =====
        # Apply threshold to both masks
        _, u2net_thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        _, modnet_thresh = cv2.threshold(modnet_mask, 128, 255, cv2.THRESH_BINARY)

        # Combine masks, prioritizing MODNet in the face region (if MODNet detects a face well)
        combined_mask = cv2.bitwise_or(u2net_thresh, modnet_thresh)

        # Refine edges using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

        # Apply Gaussian blur for smoother edges
        final_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)

        if return_mask:
            success, buf = cv2.imencode('.png', final_mask)
            if not success:
                raise ValueError("Failed to encode mask.")
            return buf.tobytes()

        # Create the final image with an alpha channel
        result_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2BGRA)
        result_image[:, :, 3] = final_mask

        # Determine the bounding box of the person
        _, thresh = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No person detected in the image")

        # Get the largest contour (the person)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the extra space for the top margin
        extra_space = int(original_h * top_margin_percent)

        # Create a new image with the extra space
        new_height = h + extra_space
        new_width = w

        # Create a blank image with an alpha channel
        final_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

        # Place the person at the bottom of the new image, leaving space at the top
        final_image[extra_space:, :] = result_image[y:y+h, x:x+w]

        # Convert the final image to bytes
        success, output_buffer = cv2.imencode(".png", final_image)
        if not success:
            raise ValueError("Failed to encode output image.")

        return output_buffer.tobytes()

    except Exception as e:
        print(f"[remove_background] Error: {e}")
        raise

class ImageSheetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Sheet Creator")

        self.image_path = None
        self.original_image = None
        self.processed_image_bytes = None

        # Styles
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 16), padding=6)
        style.configure("TLabel", font=("Arial", 16))
        style.configure("TEntry", font=("Arial", 16))

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack()
        #type name
        ttk.Label(main_frame, text="Display name : ").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(main_frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=5, padx=5)
        #select image
        self.select_button = ttk.Button(main_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=10)
        #horizontal count
        ttk.Label(main_frame, text="Horizontal count : ").grid(row=2, column=0, sticky="w")
        self.width_entry = ttk.Entry(main_frame, width=20)
        self.width_entry.grid(row=2, column=1, pady=5, padx=5)
        #vertical count
        ttk.Label(main_frame, text="Vertical count : ").grid(row=3, column=0, sticky="w")
        self.height_entry = ttk.Entry(main_frame, width=20)
        self.height_entry.grid(row=3, column=1, pady=5, padx=5)
        # #margin
        # ttk.Label(main_frame, text="(optional) Margin : ").grid(row=4, column=0, sticky="w")
        # self.margin_entry = ttk.Entry(main_frame, width=20)
        # self.margin_entry.grid(row=4, column=1, pady=5, padx=5)
        # self.margin_entry.insert(0, "50") 
        #create sheet
        self.create_button = ttk.Button(main_frame, text="Create A4 Sheet", command=self.create_sheet)
        self.create_button.grid(row=4, column=0, columnspan=2, pady=10)
        #image preview
        self.preview_label = ttk.Label(main_frame)
        self.preview_label.grid(row=5, column=0, columnspan=2, pady=10)
        # #sheet preview
        # self.sheet_preview_label = ttk.Label(main_frame)
        # self.sheet_preview_label.grid(row=0, column=3, columnspan=2, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image_path = file_path
            self.load_and_process_preview()

    def load_and_process_preview(self):
        try:
            self.processed_image_bytes = remove_background(self.image_path)
            if self.processed_image_bytes:
                np_array = np.frombuffer(self.processed_image_bytes, dtype=np.uint8)
                processed_image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
                if processed_image_cv2 is not None and processed_image_cv2.shape[2] == 4:
                    # Convert BGRA to RGBA for PIL
                    processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image_cv2, cv2.COLOR_BGRA2RGBA))
                else:
                    processed_image_pil = Image.open(BytesIO(self.processed_image_bytes)).convert("RGB")

                # Resize for preview
                h, w = processed_image_pil.size
                max_dim = max(h, w)
                scale = 300 / max_dim
                img_small = processed_image_pil.resize((int(w * scale), int(h * scale)))
                img_tk = ImageTk.PhotoImage(image=img_small)
                self.preview_label.config(image=img_tk)
                self.preview_label.image = img_tk  # keep reference
            else:
                self.preview_label.config(text="Error processing image for preview.")
        except Exception as e:
            self.preview_label.config(text=f"Error: {e}")

    def create_sheet(self):

        #check if no image
        if self.processed_image_bytes is None:
            print("No processed image available.")
            return
        #check if no name
        name = self.name_entry.get().strip()
        if not name:
            print("No name entered.")
            return
        #check if no dimensions
        grid_size_x = int(self.width_entry.get())
        grid_size_y = int(self.height_entry.get())
        if not grid_size_x or not grid_size_y:
            print("No dimensions provided")
            return
        #check if no margin
        # if not self.margin_entry:
        #     print("no margin")
        #     return
        # margin = int(self.margin_entry.get())

        #sheet setup
        sheet = np.full((A5_HEIGHT, A5_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
        np_array = np.frombuffer(self.processed_image_bytes, dtype=np.uint8)
        processed_image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
        if processed_image_cv2 is None:
            print("Error decoding processed image.")
            return

        if processed_image_cv2.shape[2] == 4:
            processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image_cv2, cv2.COLOR_BGRA2RGBA))
        else:
            processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image_cv2, cv2.COLOR_BGR2RGBA))

        # Create a new white background image with the same size
        white_background = Image.new("RGB", processed_image_pil.size, "white")

        # Make the white background the base and paste the image with transparency
        white_background.paste(processed_image_pil, mask=processed_image_pil.split()[3] if processed_image_pil.mode == 'RGBA' else None)
        processed_image_pil = white_background.convert("RGBA") # Ensure it's in RGBA for consistent pasting later

        margin = 50
        spacing_x = (A5_WIDTH - 2 * margin) / grid_size_x
        spacing_y = (A5_HEIGHT - 2 * margin) / grid_size_y

        # Resize processed image for the sheet
        image_height = int(spacing_y * 0.7)
        image_width = int(spacing_x * 0.8)
        resized_image = processed_image_pil.resize((image_width, image_height))
        resized_image_cv2 = np.array(resized_image)

        # Create a PIL Image for the sheet to draw text
        sheet_pil = Image.fromarray(sheet)
        draw = ImageDraw.Draw(sheet_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 40)  # Ensure this font supports Arabic
        except IOError:
            try:
                font = ImageFont.truetype("Amiri-Regular.ttf", 40) # Try a common Arabic font
            except IOError:
                font = ImageFont.load_default()
                print("Warning: Could not load Arial or Amiri font. Using default font.")

        reshaped_text = arabic_reshaper.reshape(name)
        bidi_text = get_display(reshaped_text)

        for row in range(grid_size_y):
            for col in range(grid_size_x):
                x_pos = int(margin + col * spacing_x + (spacing_x - image_width) / 2)
                y_pos_image = int(margin + row * spacing_y)
                y_pos_text = y_pos_image + image_height + 10

                # Paste the processed image
                sheet_pil.paste(resized_image, (x_pos, y_pos_image))

                # Draw centered text
                bbox = draw.textbbox((0, 0), bidi_text, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((x_pos + image_width // 2 - text_width // 2, y_pos_text), bidi_text, font=font, fill=TEXT_COLOR)

        # Save final sheet
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")],
                                                initialfile=self.name_entry.get(),
                                                title="Save Image Sheet As")
        if save_path:
            sheet_pil.save(save_path)
            print(f"Saved")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSheetApp(root)
    root.mainloop()