import pygame
import sys
import keras
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import backend as K


pygame.init()


CANVAS_WIDTH, CANVAS_HEIGHT = 600, 200
UI_HEIGHT = 200  # Increased from 160 to 200 for extra space
WINDOW_WIDTH, WINDOW_HEIGHT = CANVAS_WIDTH, CANVAS_HEIGHT + UI_HEIGHT

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (40, 40, 40)
ACCENT = (180, 180, 180)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Drawing App")

canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
canvas.fill(WHITE)

font = pygame.font.SysFont("Arial", 22)
small_font = pygame.font.SysFont("Arial", 16)

brush_size = 8
is_drawing = False
is_erasing = False
last_pos = None

BUTTON_W, BUTTON_H = 110, 40
BUTTON_SPACING = 30
NUM_BUTTONS = 4
TOTAL_BUTTONS_WIDTH = NUM_BUTTONS * BUTTON_W + (NUM_BUTTONS - 1) * BUTTON_SPACING
BUTTONS_Y = CANVAS_HEIGHT + 20 
BUTTONS_X = (WINDOW_WIDTH - TOTAL_BUTTONS_WIDTH) // 2

clear_btn = pygame.Rect(BUTTONS_X, BUTTONS_Y, BUTTON_W, BUTTON_H)
eraser_btn = pygame.Rect(BUTTONS_X + (BUTTON_W + BUTTON_SPACING), BUTTONS_Y, BUTTON_W, BUTTON_H)
predict_btn = pygame.Rect(BUTTONS_X + 2 * (BUTTON_W + BUTTON_SPACING), BUTTONS_Y, BUTTON_W, BUTTON_H)
save_btn = pygame.Rect(BUTTONS_X + 3 * (BUTTON_W + BUTTON_SPACING), BUTTONS_Y, BUTTON_W, BUTTON_H)

SLIDER_W, SLIDER_H = 220, 10
SLIDER_Y = BUTTONS_Y + BUTTON_H + 40  
SLIDER_X = (WINDOW_WIDTH - SLIDER_W) // 2
slider_rect = pygame.Rect(SLIDER_X, SLIDER_Y, SLIDER_W, SLIDER_H)

MIN_BRUSH, MAX_BRUSH = 2, 32

chars = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

loaded_model = keras.models.load_model('best_model.keras', custom_objects={'ctc_loss': ctc_loss})

def draw_button(rect, text, active=False):
    color = DARK_GRAY if active else ACCENT
    pygame.draw.rect(screen, color, rect, border_radius=8)
    label = font.render(text, True, WHITE if active else BLACK)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)

def draw_slider(rect, value):
    pygame.draw.rect(screen, ACCENT, rect, border_radius=4)
    percent = (value - MIN_BRUSH) / (MAX_BRUSH - MIN_BRUSH)
    knob_x = rect.x + int(percent * rect.width)
    pygame.draw.circle(screen, BLACK, (knob_x, rect.y + rect.height // 2), 14)
    label = font.render(f"Brush Size: {value}", True, BLACK)
    label_rect = label.get_rect(center=(rect.x + rect.width // 2, rect.y - 22))
    screen.blit(label, label_rect)

def save_canvas():
    small_canvas = pygame.transform.smoothscale(canvas, (300, 100))
    pygame.image.save(small_canvas, "drawing.png")

def process_image(img):
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)
    img = img / 255.0
    return img

def predict_func():
    global predicted_text
    predicted_text = ""
    try:
        save_canvas()
        image_path = "drawing.png"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image")
        
        img = process_image(img)
        img = np.expand_dims(img, axis=0)

        prediction = loaded_model.predict(img, verbose=0)
        input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
        decoded = K.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
        output_text = ''.join([chars[int(i)] for i in decoded[0] if int(i) != -1])
        predicted_text = output_text if output_text else "No prediction"
    except Exception as e:
        predicted_text = f"Error: {str(e)}"

def main():
    global brush_size, is_drawing, is_erasing, last_pos, predicted_text
    predicted_text = ""
    running = True
    while running:
        screen.fill(LIGHT_GRAY)
        pygame.draw.rect(screen, BLACK, (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), border_radius=12)
        screen.blit(canvas, (0, 0))
        pygame.draw.rect(screen, WHITE, (0, CANVAS_HEIGHT, WINDOW_WIDTH, UI_HEIGHT))
        draw_button(clear_btn, "Clear")
        draw_button(eraser_btn, "Eraser", active=is_erasing)
        draw_button(predict_btn, "Predict")
        draw_button(save_btn, "Save")
        draw_slider(slider_rect, brush_size)
        if predicted_text:
            text_label = small_font.render(f"Predicted: {predicted_text}", True, BLACK)
            text_rect = text_label.get_rect(center=(WINDOW_WIDTH // 2, SLIDER_Y + 60))  
            screen.blit(text_label, text_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if clear_btn.collidepoint(x, y):
                    canvas.fill(WHITE)
                    predicted_text = ""
                elif eraser_btn.collidepoint(x, y):
                    is_erasing = not is_erasing
                elif save_btn.collidepoint(x, y):
                    save_canvas()
                elif predict_btn.collidepoint(x, y):
                    predict_func()
                elif slider_rect.collidepoint(x, y):
                    rel_x = x - slider_rect.x
                    percent = min(max(rel_x / slider_rect.width, 0), 1)
                    brush_size = int(MIN_BRUSH + percent * (MAX_BRUSH - MIN_BRUSH))
                elif y < CANVAS_HEIGHT:
                    is_drawing = True
                    last_pos = (x, y)
            elif event.type == pygame.MOUSEBUTTONUP:
                is_drawing = False
                last_pos = None
            elif event.type == pygame.MOUSEMOTION and is_drawing:
                x, y = event.pos
                if y < CANVAS_HEIGHT:
                    color = WHITE if is_erasing else BLACK
                    if last_pos is not None:
                        pygame.draw.line(canvas, color, last_pos, (x, y), brush_size*2)
                    pygame.draw.circle(canvas, color, (x, y), brush_size)
                    last_pos = (x, y)
        pygame.display.flip()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()