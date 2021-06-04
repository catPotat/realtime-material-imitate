import os
import cv2
from screen_grab import grab_screen
from nn_backend import load_img_from_cv, transfer

CAP_LEFT = 600
CAP_TOP = 300
SQUARE_SIZE = 256
INPUT_ART_PATH = f'{os.path.dirname(os.path.realpath(__file__))}/input.png'


if __name__ == "__main__":

    input_art = cv2.imread(INPUT_ART_PATH)
    input_art = cv2.resize(input_art, dsize=(SQUARE_SIZE, SQUARE_SIZE), interpolation=cv2.INTER_CUBIC)
    window_title = f'realtime magenta arbitrary style transfer {SQUARE_SIZE}'

    print('NOW RUNNING')
    while True:
        scren_cap = grab_screen((CAP_LEFT, CAP_TOP, CAP_LEFT+SQUARE_SIZE-1, CAP_TOP+SQUARE_SIZE-1))[:,:,:3]
        preview = cv2.vconcat([input_art, scren_cap])

        stylized = transfer(load_img_from_cv(input_art),
                                load_img_from_cv(scren_cap))
        stylized = cv2.resize(stylized, dsize=(SQUARE_SIZE*2, SQUARE_SIZE*2), interpolation=cv2.INTER_LINEAR)

        to_display = cv2.hconcat([preview, stylized])
        cv2.imshow(window_title, to_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
