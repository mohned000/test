import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
import numpy as np
import os

class ImageDropApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("كاشف الصور")
        self.geometry("400x400")
        
        self.label = tk.Label(self, text="Drop Image Here", bg="lightgray", padx=10, pady=10)
        self.label.pack(fill=tk.BOTH, expand=True)
        
        self.button_select = tk.Button(self, text="حدد الصورة ", command=self.load_image)
        self.button_select.pack()
        
        self.button_predict = tk.Button(self, text="اكشف", command=self.predict_with_yolo, state=tk.DISABLED)
        self.button_predict.pack()
        
        self.file_path = None  # To store the path of the selected image
        
        # Initialize the YOLOv8 model here
        dir = os.getcwd()
        self.yolo = YOLO(dir+r'\last.pt')  # Replace with the path to your YOLOv8 model if different

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff")])
        if self.file_path:
            image = Image.open(self.file_path)
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.label.config(image=photo)
            self.label.image = photo
            self.button_predict.config(state=tk.NORMAL)  # Enable the predict button

    def predict_with_yolo(self):
        if self.file_path:
            # Perform inference
            results = self.yolo(self.file_path)
            
            # Open the image using PIL
            img_pil = Image.open(self.file_path)
            draw = ImageDraw.Draw(img_pil)
            
            # Use a default font
            font = ImageFont.load_default()
            final= []
            
            for result in results:
                #objec_detected= []
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    label = f'{self.yolo.names[cls]} {conf:.2f}'
                   # object_info = {
                                           #"xyxy": xyxy,
                                           #"confidence": conf,
                                            #"class": cls,
                                            #"label": label
                                     #}
                    #objec_detected.append(object_info["label"])
                    final.append(label)
                    #my_list = ["apple", "apple", "banana", "orange", "apple"]
                    
                    # Draw rectangle and text on the image
                    draw.rectangle([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], outline="red", width=2)
                    draw.text((xyxy[0], xyxy[1] - 10), label, fill="red", font=font)
                     
            # Convert back to ImageTk
            img_pil.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img_pil)
            self.label.config(image=photo)
            self.label.image = photo
            print(final)
            disease = [name.split()[0] for name in final]
            print(disease)  # Output: ['mohned', 'ali', 'omer']

            element_counts = {element: disease.count(element) for element in disease}
            print(element_counts)  # Output: {'apple': 3, 'banana': 1, 'orange': 1}

if __name__ == "__main__":
    app = ImageDropApp()
    app.mainloop()
