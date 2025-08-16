import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from io import BytesIO
from PIL import Image, ImageTk, ImageDraw, ImageFont
from bidi.algorithm import get_display
import arabic_reshaper
import threading

from config import A4_WIDTH, A4_HEIGHT, DPI, TEXT_COLOR
from background_removal import remove_background


class ImageSheetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Sheet Creator")
        self.image_path = None
        self.processed_image_bytes = None
        self.processed_image_pil = None
        self.loading = False

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=6)
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TEntry", font=("Arial", 12))

        main_frame = ttk.Frame(root, padding=12)
        main_frame.pack(fill="both", expand=True)

        # Row 0: Display Name
        ttk.Label(main_frame, text="Display name:").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(main_frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=6, padx=6)

        # Row 1: Font Size
        ttk.Label(main_frame, text="Font size:").grid(row=1, column=0, sticky="w")
        self.font_size_var = tk.StringVar(value="40")
        self.font_size_entry = ttk.Entry(main_frame, width=5, textvariable=self.font_size_var)
        self.font_size_entry.grid(row=1, column=1, pady=6, padx=6, sticky="w")

        # Row 2: Count
        ttk.Label(main_frame, text="Count:").grid(row=2, column=0, sticky="w")
        count_frame = ttk.Frame(main_frame)
        count_frame.grid(row=2, column=1, sticky="w", pady=6, padx=6)
        self.width_entry = ttk.Entry(count_frame, width=5)
        self.width_entry.pack(side="left")
        self.width_entry.insert(0, "4")
        ttk.Label(count_frame, text="x").pack(side="left", padx=4)
        self.height_entry = ttk.Entry(count_frame, width=5)
        self.height_entry.pack(side="left")
        self.height_entry.insert(0, "6")

        # Row 3: Add Outline
        self.outline_var = tk.BooleanVar(value=False)
        outline_frame = ttk.Frame(main_frame)
        outline_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        self.outline_check = ttk.Checkbutton(outline_frame, text="Add outline      ", variable=self.outline_var)
        self.outline_check.pack(side="left")
        self.outline_size_var = tk.StringVar(value="3")
        self.outline_size_entry = ttk.Entry(outline_frame, width=5, textvariable=self.outline_size_var)
        self.outline_size_entry.pack(side="left", padx=6)

        # Row 4: Quarter Sheet Mode
        self.quarter_var = tk.BooleanVar(value=False)
        self.quarter_check = ttk.Checkbutton(main_frame, text="Quarter-sheet mode (3×2)", variable=self.quarter_var)
        self.quarter_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=6)

        # Row 5: Select Image
        self.select_button = ttk.Button(main_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=5, column=0, columnspan=2, pady=6)

        # Row 6: Review Button
        self.review_button = ttk.Button(main_frame, text="Review Final Sheet", command=self.review_sheet)
        self.review_button.grid(row=6, column=0, columnspan=2, pady=6)

        # Row 7: Create buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=10)
        self.create_button = ttk.Button(btn_frame, text="Create A4 Sheet", command=lambda: self.start_generate_sheet(False))
        self.create_button.pack(side="left", padx=4)
        self.create_pdf_button = ttk.Button(btn_frame, text="Create PDF", command=lambda: self.start_generate_sheet(True))
        self.create_pdf_button.pack(side="left", padx=4)

        # Row 8: Preview
        self.preview_label = ttk.Label(main_frame, text="No preview", anchor="center")
        self.preview_label.grid(row=8, column=0, columnspan=2, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image_path = file_path
            self.start_loading_preview()

    def start_loading_preview(self):
        self.loading = True
        threading.Thread(target=self.load_and_process_preview, daemon=True).start()
        self.animate_loading()

    def animate_loading(self):
        if self.loading:
            current_text = self.preview_label.cget("text")
            if not current_text or "Loading" not in current_text:
                self.preview_label.config(text="Loading", image="")
            else:
                dots = (current_text.count(".") + 1) % 4
                self.preview_label.config(text="Loading" + "." * dots)
            self.root.after(500, self.animate_loading)

    def load_and_process_preview(self):
        try:
            self.processed_image_bytes = remove_background(self.image_path)
            self.processed_image_pil = Image.open(BytesIO(self.processed_image_bytes)).convert("RGBA")

            width, height = self.processed_image_pil.size
            max_dim = max(width, height)
            scale = min(300 / max_dim, 1.0)
            small = self.processed_image_pil.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

            img_tk = ImageTk.PhotoImage(small)
            self.root.after(0, lambda: self.preview_label.config(image=img_tk, text=""))
            self.root.after(0, lambda: setattr(self.preview_label, "image", img_tk))
        except Exception as ex:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"Failed to process image:\n{ex}"))
            self.root.after(0, lambda: self.preview_label.config(text="Error processing preview", image=""))
            self.root.after(0, lambda: setattr(self.preview_label, "image", None))
        finally:
            self.loading = False

    def start_generate_sheet(self, as_pdf):
        threading.Thread(target=self.generate_sheet, args=(as_pdf,), daemon=True).start()

    def generate_sheet(self, as_pdf=False):
        if self.processed_image_bytes is None or self.processed_image_pil is None:
            self.root.after(0, lambda: messagebox.showerror("No Image", "Please select and process an image first."))
            return

        name = self.name_entry.get().strip()
        if not name:
            self.root.after(0, lambda: messagebox.showerror("No Name", "Please enter a display name."))
            return

        quarter_mode = self.quarter_var.get()

        try:
            if quarter_mode:
                grid_x, grid_y = 3, 2
            else:
                grid_x = int(self.width_entry.get())
                grid_y = int(self.height_entry.get())
                if grid_x <= 0 or grid_y <= 0:
                    raise ValueError()
        except Exception:
            self.root.after(0, lambda: messagebox.showerror("Invalid Grid", "Horizontal and vertical counts must be positive integers."))
            return

        font_size = int(self.font_size_var.get() or 40)
        outline_size = int(self.outline_size_var.get()) if self.outline_var.get() else 0
        tile_padding = 18

        try:
            if quarter_mode:
                quarter_w = A4_WIDTH // 2
                quarter_h = A4_HEIGHT // 2
                work_width = quarter_h
                work_height = quarter_w

                sheet_quarter = Image.new("RGB", (work_width, work_height), "white")
                draw = ImageDraw.Draw(sheet_quarter)

                margin = 0
                spacing_x = int((work_width - 2 * margin) / grid_x)
                spacing_y = int((work_height - 2 * margin) / grid_y)

                image_height = max(10, int(spacing_y * 0.7))
                image_width = max(10, int(spacing_x * 0.8))

                resized_image = self.processed_image_pil.resize((image_width, image_height), Image.LANCZOS)

                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("Amiri-Regular.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()
                        self.root.after(0, lambda: messagebox.showwarning("Font Warning", "Could not load Arial or Amiri font. Using default font."))

                reshaped_text = arabic_reshaper.reshape(name)
                bidi_text = get_display(reshaped_text)

                for row in range(grid_y):
                    for col in range(grid_x):
                        x_pos = margin + col * spacing_x + max(0, (spacing_x - image_width) // 2)
                        y_pos_image = margin + row * spacing_y
                        y_pos_text = y_pos_image + image_height + 8

                        sheet_quarter.paste(resized_image, (x_pos, y_pos_image), resized_image if resized_image.mode == "RGBA" else None)

                        bbox_for_measure = draw.textbbox((0, 0), bidi_text, font=font)
                        text_width = bbox_for_measure[2] - bbox_for_measure[0]
                        text_height = bbox_for_measure[3] - bbox_for_measure[1]
                        text_x = x_pos + image_width // 2 - text_width // 2

                        draw.text((text_x, y_pos_text), bidi_text, font=font, fill=TEXT_COLOR)

                        if outline_size > 0:
                            left = max(0, x_pos - tile_padding)
                            top = max(0, y_pos_image - tile_padding)
                            right = min(work_width, x_pos + image_width + tile_padding)
                            bottom = min(work_height, y_pos_text + text_height + tile_padding)
                            draw.rectangle([left, top, right, bottom], outline="black", width=max(1, outline_size))

                # Outline around quarter
                draw.rectangle([0, 0, work_width - 1, work_height - 1], outline="black", width=3)

                sheet_rotated = sheet_quarter.rotate(90, expand=True)

                full_sheet = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), "white")
                full_sheet.paste(sheet_rotated, (0, 0))
                sheet = full_sheet

            else:
                work_width, work_height = A4_WIDTH, A4_HEIGHT
                sheet = Image.new("RGB", (work_width, work_height), "white")
                draw = ImageDraw.Draw(sheet)

                margin = 0
                spacing_x = int((work_width - 2 * margin) / grid_x)
                spacing_y = int((work_height - 2 * margin) / grid_y)

                image_height = max(10, int(spacing_y * 0.7))
                image_width = max(10, int(spacing_x * 0.8))

                resized_image = self.processed_image_pil.resize((image_width, image_height), Image.LANCZOS)

                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("Amiri-Regular.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()
                        self.root.after(0, lambda: messagebox.showwarning("Font Warning", "Could not load Arial or Amiri font. Using default font."))

                reshaped_text = arabic_reshaper.reshape(name)
                bidi_text = get_display(reshaped_text)

                for row in range(grid_y):
                    for col in range(grid_x):
                        x_pos = margin + col * spacing_x + max(0, (spacing_x - image_width) // 2)
                        y_pos_image = margin + row * spacing_y
                        y_pos_text = y_pos_image + image_height + 8

                        sheet.paste(resized_image, (x_pos, y_pos_image), resized_image if resized_image.mode == "RGBA" else None)

                        bbox_for_measure = draw.textbbox((0, 0), bidi_text, font=font)
                        text_width = bbox_for_measure[2] - bbox_for_measure[0]
                        text_height = bbox_for_measure[3] - bbox_for_measure[1]
                        text_x = x_pos + image_width // 2 - text_width // 2

                        draw.text((text_x, y_pos_text), bidi_text, font=font, fill=TEXT_COLOR)

                        if outline_size > 0:
                            left = max(0, x_pos - tile_padding)
                            top = max(0, y_pos_image - tile_padding)
                            right = min(work_width, x_pos + image_width + tile_padding)
                            bottom = min(work_height, y_pos_text + text_height + tile_padding)
                            draw.rectangle([left, top, right, bottom], outline="black", width=max(1, outline_size))

                # Outline around full sheet
                draw.rectangle([0, 0, work_width - 1, work_height - 1], outline="black", width=3)

            # save
            if as_pdf:
                save_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                         filetypes=[("PDF files", "*.pdf")],
                                                         initialfile=self.name_entry.get(),
                                                         title="Save PDF As")
                if save_path:
                    sheet.save(save_path, "PDF", resolution=DPI)
                    self.root.after(0, lambda: messagebox.showinfo("Saved", f"PDF saved to:\n{save_path}"))
            else:
                save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                         filetypes=[("PNG files", "*.png")],
                                                         initialfile=self.name_entry.get(),
                                                         title="Save Image Sheet As")
                if save_path:
                    sheet.save(save_path, dpi=(DPI, DPI))
                    self.root.after(0, lambda: messagebox.showinfo("Saved", f"Sheet saved to:\n{save_path}"))

        except Exception as ex:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to create sheet:\n{ex}"))

    def review_sheet(self):
        # This method creates the full sheet and shows it in a new window for review without saving
        if self.processed_image_bytes is None or self.processed_image_pil is None:
            messagebox.showerror("No Image", "Please select and process an image first.")
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("No Name", "Please enter a display name.")
            return

        quarter_mode = self.quarter_var.get()

        try:
            if quarter_mode:
                grid_x, grid_y = 3, 2
            else:
                grid_x = int(self.width_entry.get())
                grid_y = int(self.height_entry.get())
                if grid_x <= 0 or grid_y <= 0:
                    raise ValueError()
        except Exception:
            messagebox.showerror("Invalid Grid", "Horizontal and vertical counts must be positive integers.")
            return

        font_size = int(self.font_size_var.get() or 40)
        outline_size = int(self.outline_size_var.get()) if self.outline_var.get() else 0
        tile_padding = 18

        try:
            if quarter_mode:
                quarter_w = A4_WIDTH // 2
                quarter_h = A4_HEIGHT // 2
                work_width = quarter_h
                work_height = quarter_w

                sheet_quarter = Image.new("RGB", (work_width, work_height), "white")
                draw = ImageDraw.Draw(sheet_quarter)

                margin = 0
                spacing_x = int((work_width - 2 * margin) / grid_x)
                spacing_y = int((work_height - 2 * margin) / grid_y)

                image_height = max(10, int(spacing_y * 0.7))
                image_width = max(10, int(spacing_x * 0.8))

                resized_image = self.processed_image_pil.resize((image_width, image_height), Image.LANCZOS)

                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("Amiri-Regular.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()

                reshaped_text = arabic_reshaper.reshape(name)
                bidi_text = get_display(reshaped_text)

                for row in range(grid_y):
                    for col in range(grid_x):
                        x_pos = margin + col * spacing_x + max(0, (spacing_x - image_width) // 2)
                        y_pos_image = margin + row * spacing_y
                        y_pos_text = y_pos_image + image_height + 8

                        sheet_quarter.paste(resized_image, (x_pos, y_pos_image), resized_image if resized_image.mode == "RGBA" else None)

                        bbox_for_measure = draw.textbbox((0, 0), bidi_text, font=font)
                        text_width = bbox_for_measure[2] - bbox_for_measure[0]
                        text_height = bbox_for_measure[3] - bbox_for_measure[1]
                        text_x = x_pos + image_width // 2 - text_width // 2

                        draw.text((text_x, y_pos_text), bidi_text, font=font, fill=TEXT_COLOR)

                        if outline_size > 0:
                            left = max(0, x_pos - tile_padding)
                            top = max(0, y_pos_image - tile_padding)
                            right = min(work_width, x_pos + image_width + tile_padding)
                            bottom = min(work_height, y_pos_text + text_height + tile_padding)
                            draw.rectangle([left, top, right, bottom], outline="black", width=max(1, outline_size))

                draw.rectangle([0, 0, work_width - 1, work_height - 1], outline="black", width=3)

                sheet_rotated = sheet_quarter.rotate(90, expand=True)
                full_sheet = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), "white")
                full_sheet.paste(sheet_rotated, (0, 0))
                sheet = full_sheet

            else:
                work_width, work_height = A4_WIDTH, A4_HEIGHT
                sheet = Image.new("RGB", (work_width, work_height), "white")
                draw = ImageDraw.Draw(sheet)

                margin = 0
                spacing_x = int((work_width - 2 * margin) / grid_x)
                spacing_y = int((work_height - 2 * margin) / grid_y)

                image_height = max(10, int(spacing_y * 0.7))
                image_width = max(10, int(spacing_x * 0.8))

                resized_image = self.processed_image_pil.resize((image_width, image_height), Image.LANCZOS)

                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try:
                        font = ImageFont.truetype("Amiri-Regular.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()

                reshaped_text = arabic_reshaper.reshape(name)
                bidi_text = get_display(reshaped_text)

                for row in range(grid_y):
                    for col in range(grid_x):
                        x_pos = margin + col * spacing_x + max(0, (spacing_x - image_width) // 2)
                        y_pos_image = margin + row * spacing_y
                        y_pos_text = y_pos_image + image_height + 8

                        sheet.paste(resized_image, (x_pos, y_pos_image), resized_image if resized_image.mode == "RGBA" else None)

                        bbox_for_measure = draw.textbbox((0, 0), bidi_text, font=font)
                        text_width = bbox_for_measure[2] - bbox_for_measure[0]
                        text_height = bbox_for_measure[3] - bbox_for_measure[1]
                        text_x = x_pos + image_width // 2 - text_width // 2

                        draw.text((text_x, y_pos_text), bidi_text, font=font, fill=TEXT_COLOR)

                        if outline_size > 0:
                            left = max(0, x_pos - tile_padding)
                            top = max(0, y_pos_image - tile_padding)
                            right = min(work_width, x_pos + image_width + tile_padding)
                            bottom = min(work_height, y_pos_text + text_height + tile_padding)
                            draw.rectangle([left, top, right, bottom], outline="black", width=max(1, outline_size))

                draw.rectangle([0, 0, work_width - 1, work_height - 1], outline="black", width=3)

            preview_win = tk.Toplevel(self.root)
            preview_win.title("Final Sheet Preview")

            max_preview_size = 800
            scale = min(max_preview_size / A4_WIDTH, max_preview_size / A4_HEIGHT, 1.0)
            preview_img = sheet.resize((int(A4_WIDTH * scale), int(A4_HEIGHT * scale)), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(preview_img)

            label = ttk.Label(preview_win, image=img_tk)
            label.image = img_tk
            label.pack()

        except Exception as ex:
            messagebox.showerror("Error", f"Failed to generate preview:\n{ex}")
