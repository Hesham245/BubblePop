import cv2
import mediapipe as mp
import pygame
import random
import numpy as np
from pygame import gfxdraw
import os
import json
from datetime import datetime

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1280, 720))

# Environment settings
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

# Game constants
LEADERBOARD_FILE = "leaderboard.json"
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
BUTTON_COLOR = (100, 200, 100)
BUTTON_HOVER_COLOR = (50, 150, 50)
TEXT_COLOR = (255, 255, 255)
MENU_COLOR = (30, 136, 229)
INITIAL_TIME = 10
LIVES = 3

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_draw = mp.solutions.drawing_utils

# Audio setup
pygame.mixer.init()
pop_sound = pygame.mixer.Sound('pop.wav')
bomb_sound = pygame.mixer.Sound('bomb.wav')
pygame.mixer.music.load('background.mp3')
pygame.mixer.music.set_volume(0.3)
pygame.mixer.music.play(-1)

# Load images
bg_image = cv2.imread('spongebob_bg.jpg')
bg_image = cv2.resize(bg_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
bomb_image = pygame.image.load('bomb.png').convert_alpha()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)


# Modified Bubble class
class Bubble:
    def __init__(self, is_bomb=False):
        # Load images FIRST before creating surface
        self.bubble_img = pygame.image.load('bubble.png').convert_alpha()
        self.bomb_img = pygame.image.load('bomb.png').convert_alpha()

        # Then initialize other properties
        self.radius = random.randint(40, 60)
        self.x = random.randint(self.radius, WINDOW_WIDTH - self.radius)
        self.y = random.randint(self.radius, WINDOW_HEIGHT - self.radius)
        self.is_bomb = is_bomb
        self.dx = random.uniform(-4, 4)
        self.dy = random.uniform(-4, 4)
        self.surface = self.create_bubble_surface()

    def create_bubble_surface(self):
        diameter = self.radius * 2

        if self.is_bomb:
            img = pygame.transform.scale(self.bomb_img, (diameter, diameter))
        else:
            img = pygame.transform.scale(self.bubble_img, (diameter, diameter))
            alpha = random.randint(150, 220)
            img.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)

        surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        surface.blit(img, (0, 0))
        return surface

    def update_position(self):
        self.x += self.dx
        self.y += self.dy

        if self.x < self.radius or self.x > WINDOW_WIDTH - self.radius:
            self.dx *= -1
        if self.y < self.radius or self.y > WINDOW_HEIGHT - self.radius:
            self.dy *= -1

    def draw(self):
        self.update_position()
        screen.blit(self.surface, (self.x - self.radius, self.y - self.radius))


def draw_button(text, position, width=300, height=60):
    font = pygame.font.Font(None, 50)
    text_render = font.render(text, True, TEXT_COLOR)
    rect = pygame.Rect(0, 0, width, height)
    rect.center = position

    mouse_pos = pygame.mouse.get_pos()
    clicked = False

    if rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, BUTTON_HOVER_COLOR, rect, border_radius=25)
        if pygame.mouse.get_pressed()[0]:
            clicked = True
    else:
        pygame.draw.rect(screen, BUTTON_COLOR, rect, border_radius=25)

    screen.blit(text_render, text_render.get_rect(center=rect.center))
    return clicked


def get_player_name():
    name = ""
    input_active = True
    font = pygame.font.Font(None, 74)

    while input_active:
        screen.fill(MENU_COLOR)
        prompt_text = font.render("ENTER YOUR NAME:", True, (255, 255, 255))
        screen.blit(prompt_text, (WINDOW_WIDTH // 2 - prompt_text.get_width() // 2, 200))

        name_text = font.render(name, True, (255, 255, 0))
        screen.blit(name_text, (WINDOW_WIDTH // 2 - name_text.get_width() // 2, 300))

        draw_button("CONTINUE", (WINDOW_WIDTH // 2, 450), 400, 80)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and len(name) > 0:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                elif event.unicode.isalpha() and len(name) < 12:
                    name += event.unicode.upper()

    return name.upper()


def show_instructions():
    instructions = [
        "HOW TO PLAY:",
        "1. You start with 10 seconds and 3 lives",
        "2. Pop green bubbles (+) to gain time",
        "3. Avoid red bomb bubbles (💣) - they remove lives!",
        "4. Each + bubble gives +2 seconds",
        "5. Each bomb 💣 costs 1 life",
        "6. Survive as long as possible!"
    ]

    while True:
        screen.fill(MENU_COLOR)
        title_font = pygame.font.Font(None, 80)
        title_text = title_font.render("Instructions", True, (255, 255, 255))
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 50))

        y = 150
        font = pygame.font.Font(None, 40)
        for line in instructions:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
            y += 50

        if draw_button("Back to Menu", (WINDOW_WIDTH // 2, 600), 300, 60):
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()

        pygame.display.update()


def save_to_leaderboard(name, score):
    try:
        with open(LEADERBOARD_FILE, 'r') as f:
            leaderboard = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        leaderboard = []

    leaderboard.append({
        "name": name,
        "score": round(score, 2),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    leaderboard = leaderboard[:10]

    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f, indent=2)


def show_leaderboard():
    try:
        with open(LEADERBOARD_FILE, 'r') as f:
            leaderboard = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        leaderboard = []

    while True:
        screen.fill(MENU_COLOR)
        title_font = pygame.font.Font(None, 80)
        title_text = title_font.render("Leaderboard", True, (255, 255, 255))
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 50))

        y = 150
        font = pygame.font.Font(None, 40)
        for idx, entry in enumerate(leaderboard[:10]):
            text = f"{idx + 1}. {entry['name']} - {entry['score']}s ({entry['date']})"
            entry_text = font.render(text, True, (255, 255, 255))
            screen.blit(entry_text, (100, y))
            y += 50

        if draw_button("Back", (WINDOW_WIDTH // 2, 600), 200, 50):
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()

        pygame.display.update()


def main_menu():
    while True:
        screen.fill(MENU_COLOR)
        title_font = pygame.font.Font(None, 100)
        title_text = title_font.render("BUBBLE CHALLENGE!", True, (255, 255, 0))
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 100))

        button_y = 300
        button_spacing = 100

        if draw_button("START GAME", (WINDOW_WIDTH // 2, button_y), 400, 80):
            game_loop()

        if draw_button("HOW TO PLAY", (WINDOW_WIDTH // 2, button_y + button_spacing), 400, 80):
            show_instructions()

        if draw_button("LEADERBOARD", (WINDOW_WIDTH // 2, button_y + 2 * button_spacing), 400, 80):
            show_leaderboard()

        if draw_button("QUIT GAME", (WINDOW_WIDTH // 2, button_y + 3 * button_spacing), 400, 80):
            pygame.quit()
            cap.release()
            exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()

        pygame.display.update()


def game_loop():
    bubbles = []
    spawn_rate = 45
    max_bubbles = 15
    game_active = True
    font = pygame.font.Font(None, 74)
    clock = pygame.time.Clock()

    start_time = pygame.time.get_ticks()
    current_time = INITIAL_TIME
    lives = LIVES
    score = 0
    saved = False
    final_score = 0

    while True:
        dt = clock.tick(60) / 1000
        time_elapsed = (pygame.time.get_ticks() - start_time) / 1000
        current_time = INITIAL_TIME + score - time_elapsed

        # Game over conditions
        if (current_time <= 0 or lives <= 0) and game_active:
            game_active = False
            final_score = time_elapsed + score
            if current_time > 0 and not saved:
                player_name = get_player_name()
                save_to_leaderboard(player_name, final_score)
                saved = True

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()

        # Camera processing
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_seg = selfie_segmentation.process(rgb_frame)
        condition = np.stack((results_seg.segmentation_mask,) * 3, axis=-1) > 0.2
        output_frame = np.where(condition, rgb_frame, bg_image)

        # Hand tracking
        results = hands.process(rgb_frame)
        index_finger_pos = None
        if results.multi_hand_landmarks and game_active:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = WINDOW_WIDTH - int(index_tip.x * WINDOW_WIDTH)
            y = int(index_tip.y * WINDOW_HEIGHT)
            index_finger_pos = (x, y)
            mp_draw.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert camera feed to Pygame surface
        frame = pygame.surfarray.make_surface(np.rot90(output_frame))
        screen.blit(frame, (0, 0))

        if game_active:
            # Spawn bubbles
            if len(bubbles) < max_bubbles and random.randint(1, spawn_rate) == 1:
                is_bomb = random.random() < 0.25
                bubbles.append(Bubble(is_bomb))

            # Check collisions
            if index_finger_pos:
                for bubble in bubbles[:]:
                    distance = ((bubble.x - index_finger_pos[0]) ** 2 +
                                (bubble.y - index_finger_pos[1]) ** 2) ** 0.5
                    if distance < bubble.radius:
                        if bubble.is_bomb:
                            lives -= 1
                            bomb_sound.play()
                        else:
                            score += 2
                            pop_sound.play()
                        bubbles.remove(bubble)

            # Draw bubbles
            for bubble in bubbles:
                bubble.draw()

            # Draw UI elements
            time_text = font.render(f"TIME: {max(0, current_time):.1f}s", True, (255, 255, 0))
            lives_text = font.render(f"LIVES: {lives}", True, (255, 50, 50))
            score_text = font.render(f"SCORE: {score:.1f}", True, (50, 255, 50))

            screen.blit(time_text, (20, 20))
            screen.blit(lives_text, (20, 100))
            screen.blit(score_text, (20, 180))
        else:
            # Game over screen
            result_text = font.render(f"FINAL SCORE: {final_score:.1f}s", True, (255, 255, 0))
            screen.blit(result_text, (WINDOW_WIDTH // 2 - result_text.get_width() // 2, 200))

            if draw_button("PLAY AGAIN", (WINDOW_WIDTH // 2, 350), 400, 80):
                game_loop()
                return

            if draw_button("MAIN MENU", (WINDOW_WIDTH // 2, 450), 400, 80):
                return

            if draw_button("LEADERBOARD", (WINDOW_WIDTH // 2, 550), 400, 80):
                show_leaderboard()

            if draw_button("QUIT GAME", (WINDOW_WIDTH // 2, 650), 400, 80):
                pygame.quit()
                cap.release()
                exit()

        pygame.display.update()


if __name__ == "__main__":
    main_menu()