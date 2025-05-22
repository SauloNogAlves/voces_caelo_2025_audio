import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay
import random

# Caminho da imagem neutra base
IMAGE_PATH = "rostos_base/rosto_padrao.png"

# Define a resolução da tela (ajuste conforme seu monitor)
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Carrega imagem base em escala de cinza
img_color = cv2.imread(IMAGE_PATH)
gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_base = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
h, w = img_base.shape[:2]

# Inicializa MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Detecta pontos faciais
rgb_image = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

if not results.multi_face_landmarks:
    print("Nenhum rosto detectado.")
    exit()

# Extrai 468 landmarks
landmarks = results.multi_face_landmarks[0]
points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

# Cria triangulação de Delaunay
points_np = np.array(points)
tri = Delaunay(points_np)

# Inicializa janela em tela cheia
cv2.namedWindow("Visionomia", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Visionomia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Visionomia refinada rodando... Pressione ESC para sair.")

while True:
    # Nova imagem preta para desenhar
    frame = np.zeros_like(img_base)

    # Desenha os triângulos com distorções leves
    for simplex in tri.simplices:
        pts = points_np[simplex]
        dist = np.random.randint(-3, 3, pts.shape)
        pts_dist = pts + dist

        mask = np.zeros(img_base.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_dist, 255)

        color = cv2.mean(img_base, mask=mask)[:3]
        color = tuple(map(int, color))
        cv2.fillConvexPoly(frame, pts_dist, color)

    # Ajusta proporção da imagem para tela sem deformar
    img_h, img_w = frame.shape[:2]
    scale = min(SCREEN_WIDTH / img_w, SCREEN_HEIGHT / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Cria tela preta e centraliza o rosto
    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    x_offset = (SCREEN_WIDTH - new_w) // 2
    y_offset = (SCREEN_HEIGHT - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Exibe a imagem
    cv2.imshow("Visionomia", canvas)

    if cv2.waitKey(30) == 27:  # ESC
        break

cv2.destroyAllWindows()
