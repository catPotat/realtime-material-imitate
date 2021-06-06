import numpy as np
import cv2
from screen_grab import grab_screen
from nn_backend import load_img_from_cv, transfer
import time

CAP_LEFT = 600
CAP_TOP = 300
SQUARE_SIZE = 256


if __name__ == "__main__":

    window_title = f'realtime magenta arbitrary style transfer {SQUARE_SIZE}'
    capturing = 0
    content = style = np.zeros((SQUARE_SIZE, SQUARE_SIZE, 3), dtype=np.uint8)
    print('NOW RUNNING')
    while True:
        scren_cap = grab_screen((CAP_LEFT, CAP_TOP, CAP_LEFT+SQUARE_SIZE-1, CAP_TOP+SQUARE_SIZE-1))[:,:,:3]

        if capturing == 0:
            content = scren_cap
        elif capturing == 1:
            style = scren_cap
        preview = cv2.vconcat([content, style])

        stylized = transfer(load_img_from_cv(content),
                                load_img_from_cv(style))
        stylized = cv2.resize(stylized, dsize=(SQUARE_SIZE*2, SQUARE_SIZE*2), interpolation=cv2.INTER_LINEAR)

        to_display = cv2.hconcat([preview, stylized])
        to_display = cv2.putText(to_display, 'Active', (5, SQUARE_SIZE*capturing+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), )
        cv2.imshow(window_title, to_display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            capturing = 0 if capturing else 1
        elif k == ord('e'):
            cv2.imwrite(f'{time.time()}-output.png', stylized)
            print("EXPORTED")

    cv2.destroyAllWindows()
