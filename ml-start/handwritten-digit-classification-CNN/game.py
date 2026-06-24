from tokenize import Number
from numpy import testing
from numpy.lib.type_check import imag
from tensorflow.python.keras.backend import constant
import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

BOUNDRYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

# Model of Neural Network
MODEL = load_model("bestmodel.h5")

LABELS = {0: "0", 1: "1",
          2: "2", 3: "3",
          4: "4", 5: "5",
          6: "6", 7: "7",
          8: "8", 9: "9"}

# Initialize our pygame
pygame.init()

FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
WHITE_INT = DISPLAYSURF.map_rgb(WHITE)
pygame.display.set_caption("Digit Board")

iswriting = False
number_x_coord = []
number_y_coord = []
imag_cnt = 1

PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            x_coord, y_coord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x_coord, y_coord), 4, 0)
            number_x_coord.append(x_coord)
            number_y_coord.append(y_coord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_x_coord = sorted(number_x_coord)
            number_y_coord = sorted(number_y_coord)

            rect_min_x, rect_max_x = max(number_x_coord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX,
                                                                                 number_x_coord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_y_coord[0] - BOUNDRYINC, 0), min(number_y_coord[-1] + BOUNDRYINC,
                                                                                 WINDOWSIZEX)

            number_x_coord = []
            number_y_coord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x: rect_max_x, rect_min_y: rect_max_y]\
                .T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                imag_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), "constant", constant_values=0)
                image = cv2.resize(image, (28, 28)) / WHITE_INT

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = testing.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRecObj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()
