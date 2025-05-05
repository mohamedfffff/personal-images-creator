import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from bidi.algorithm import get_display
import arabic_reshaper

# Constants
A5_WIDTH = 1748
A5_HEIGHT = 2480
BACKGROUND_COLOR = (255, 255, 255)  # white
TEXT_COLOR = (0, 0, 0)  # black

class ImageSheetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Sheet Creator")

        self.image_path = None
        self.original_image = None

        # Styles
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=6)
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TEntry", font=("Arial", 12))

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack()

        ttk.Label(main_frame, text="Name to display:").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(main_frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=5, padx=5)

        self.select_button = ttk.Button(main_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.choice_var = tk.IntVar(value=1)
        choices_frame = ttk.Frame(main_frame)
        choices_frame.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Radiobutton(choices_frame, text="Single Image", variable=self.choice_var, value=1).pack(side="left", padx=5)
        ttk.Radiobutton(choices_frame, text="4 Images", variable=self.choice_var, value=4).pack(side="left", padx=5)
        ttk.Radiobutton(choices_frame, text="9 Images", variable=self.choice_var, value=9).pack(side="left", padx=5)

        self.create_button = ttk.Button(main_frame, text="Create Sheet", command=self.create_sheet)
        self.create_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.preview_label = ttk.Label(main_frame)
        self.preview_label.grid(row=4, column=0, columnspan=2, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image_path = file_path
            self.load_and_preview()

    def load_and_preview(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.original_image = img

        # Resize for preview
        h, w = img.shape[:2]
        max_dim = max(h, w)
        scale = 300 / max_dim
        img_small = cv2.resize(img, (int(w * scale), int(h * scale)))

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_small))
        self.preview_label.config(image=img_tk)
        self.preview_label.image = img_tk  # keep reference

    def create_sheet(self):
        if self.original_image is None:
            print("No image selected.")
            return
        name = self.name_entry.get().strip()
        if not name:
            print("No name entered.")
            return

        count = self.choice_var.get()

        sheet = np.full((A5_HEIGHT, A5_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

        grid_size = 1 if count == 1 else int(count ** 0.5)
        margin = 50
        spacing_x = (A5_WIDTH - 2 * margin) / grid_size
        spacing_y = (A5_HEIGHT - 2 * margin) / grid_size

        # Resize original
        image_height = int(spacing_y * 0.7)
        image_width = int(spacing_x * 0.8)

        resized = cv2.resize(self.original_image, (image_width, image_height), interpolation=cv2.INTER_AREA)

        for row in range(grid_size):
            for col in range(grid_size):
                x = int(margin + col * spacing_x + (spacing_x - image_width) / 2)
                y = int(margin + row * spacing_y)

                sheet[y:y+image_height, x:x+image_width] = resized

        # Save temporarily
        temp_image = Image.fromarray(sheet)
        draw = ImageDraw.Draw(temp_image)

        try:
            font = ImageFont.truetype("arial.ttf", 40)  # Ensure this font supports Arabic
        except IOError:
            try:
                font = ImageFont.truetype(" Amiri-Regular.ttf", 40) # Try a common Arabic font (you might need to download this)
            except IOError:
                font = ImageFont.load_default()
                print("Warning: Could not load Arial or Amiri font. Using default font which might not support Arabic.")

        reshaped_text = arabic_reshaper.reshape(name)
        bidi_text = get_display(reshaped_text)

        for row in range(grid_size):
            for col in range(grid_size):
                x = int(margin + col * spacing_x + (spacing_x - image_width) / 2)
                y = int(margin + row * spacing_y)
                text_x = x + image_width // 2
                text_y = y + image_height + 10

                # Draw centered text
                bbox = draw.textbbox((0, 0), bidi_text, font=font)
                text_width = bbox[2] - bbox[0]

                draw.text((text_x - text_width // 2, text_y), bidi_text, font=font, fill=TEXT_COLOR)

        # Save final sheet
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")],
                                                title="Save Image Sheet As")
        if save_path:
            temp_image.save(save_path)
            print(f"Saved")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSheetApp(root)
    root.mainloop()
