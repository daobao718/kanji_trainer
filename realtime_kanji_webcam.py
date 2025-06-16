import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageDraw, ImageFont

#constants
MODEL_PATH = 'best_kanji_model.keras'
IMAGE_SIZE = (28, 28)
NUM_KANJI_CLASSES = 10
KUZUSHIJI_LABELS = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'り', 'を']
FONT_PATH = "NotoSansJP-VariableFont_wght.ttf" 
FONT_SIZE = 25

print("Starting...")
#model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' exists and is valid.")
    exit()

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print(f"Error: Could not load font from {FONT_PATH}.")
    font = ImageFont.load_default()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not find webcam.")
    exit()

print("Webcam opened successfully. Place a character inside the green box. Press 'q' to quit.")
#Constants for webcam
MIN_CONTOUR_AREA = 150 
CONFIDENCE_THRESHOLD = 80

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    # frame = cv2.flip(frame, 1) 
    frame_h, frame_w, _ = frame.shape
    
    #ROI
    roi_width = 300 
    roi_height = 300
    x1 = (frame_w - roi_width) // 2
    y1 = (frame_h - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    roi = frame[y1:y2, x1:x2]

    #processed cam
    processed_digit_display = np.zeros(IMAGE_SIZE, dtype=np.uint8) 

    if roi.size == 0:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((x1, y1 - FONT_SIZE - 5 if y1 - FONT_SIZE - 5 >= 0 else 0), 
                   "ROI Empty!", font=font, fill=(255, 0, 0, 0)) 
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow('Real-time Kanji Recognition', frame)
        cv2.imshow('Processed Digit', processed_digit_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    #preprocess ROI
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    _, thresh_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_label_text = "N/A"
    confidence = 0.0
    display_color = (0, 0, 255) 
    found_valid_character = False
    best_square_digit = np.zeros(IMAGE_SIZE, dtype=np.uint8) 

    if len(contours) > 0:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA: 
                continue
            x, y, w, h = cv2.boundingRect(contour)

            if w == 0 or h == 0:
                continue
            digit_roi_extracted = thresh_roi[y:y+h, x:x+w]
            current_square_digit = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            scale = min( (IMAGE_SIZE[0]-8) / w, (IMAGE_SIZE[1]-8) / h) 
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w == 0 or new_h == 0:
                continue
            resized_digit = cv2.resize(digit_roi_extracted, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            pad_x = (IMAGE_SIZE[0] - new_w) // 2
            pad_y = (IMAGE_SIZE[1] - new_h) // 2
            current_square_digit[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_digit

            input_image = current_square_digit.astype('float32') / 255.0
            input_image = np.expand_dims(input_image, axis=0) 
            input_image = np.expand_dims(input_image, axis=-1) 
            
            predictions = model.predict(input_image, verbose=0)
            current_predicted_class = np.argmax(predictions[0])
            current_confidence = np.max(predictions[0]) * 100

            if current_confidence >= CONFIDENCE_THRESHOLD:
                predicted_class = current_predicted_class
                confidence = current_confidence
                predicted_label_text = KUZUSHIJI_LABELS[predicted_class]
                display_color = (0, 255, 0)
                found_valid_character = True
                best_square_digit = current_square_digit 
                break

    #dipsplay
    if not found_valid_character:
        display_text = "No character detected!"
        display_color = (0, 0, 255) 
       
        processed_digit_display = np.zeros(IMAGE_SIZE, dtype=np.uint8) 
    else:
        display_text = f"Pred: {predicted_label_text} ({confidence:.2f}%)"
        processed_digit_display = best_square_digit 

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    text_x = x1
    text_y = y1 - FONT_SIZE - 5
    if text_y < 0:
        text_y = 0

    draw.text((text_x, text_y), display_text, font=font, fill=display_color + (0,))

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('Real-time Kanji Recognition', frame)
    cv2.imshow('Processed Digit', cv2.resize(processed_digit_display, (200, 200), interpolation=cv2.INTER_NEAREST))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("\nReal-time recognition stopped.")