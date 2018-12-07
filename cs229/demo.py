import cv2
import numpy as np
from time import perf_counter

from cs229.display import open_window, show_image
from cs229.files import open_video
from cs229.image import img_to_mask
from cs229.full_img import FullImage
from cs229.train_contour import ContourPredictor
from cs229.train_orientation import PosePredictor

def bound(pt, img):
    def clip(dim):
        return np.clip(np.round(pt[dim]), 0, img.shape[1-dim] - 1).astype(int)

    return (clip(0), clip(1))

def main(arrow_length=40, profile=False):
    # prepare video
    cap, props = open_video('test5')
    _, img = cap.read()
    mask = img_to_mask(img)

    # load predictors
    contour_predictor = ContourPredictor()
    pose_predictor = {type: PosePredictor(type) for type in ['male', 'female']}

    # display-specific actions
    if not profile:
        open_window()
        colors = {'female': (255, 0, 0), 'male': (0, 0, 255), 'both': (255, 0, 255), 'neither': None}

    frames = 0
    tick = perf_counter()

    while True:
        # read frame
        ok, img = cap.read()
        if not ok:
            break

        frames += 1

        # extract contours
        full_img = FullImage(img, mask=mask)

        # draw contours with a color associated with the class
        results = {'male': [], 'female': [], 'both': [], 'neither': []}

        # predict the contour label, center, and angle
        for contour in full_img.contours:
            label = contour_predictor.predict(contour)
            result = {'contour': contour}

            if label in ['male', 'female']:
                (cx, cy), angle = pose_predictor[label].predict(full_img.img, contour)
                result.update(dict(cx=cx, cy=cy, angle=angle))

            results[label].append(result)

        # when profiling, skip the drawing steps
        if profile:
            continue

        # illustrate the results
        out = img.copy()
        for label in ['male', 'female', 'both']:
            # draw outline
            contours = [result['contour'] for result in results[label]]
            cv2.drawContours(out, contours, -1, colors[label], 3)

            if label not in ['male', 'female']:
                continue

            for result in results[label]:
                # draw center
                center = bound((result['cx'], result['cy']), out)
                cv2.circle(out, center, 5, colors[label], -1)

                # draw arrow in direction of orientation
                ax = result['cx'] + arrow_length*np.cos(result['angle'])
                ay = result['cy'] - arrow_length*np.sin(result['angle'])
                tip = bound((ax, ay), out)
                cv2.arrowedLine(out, center, tip, colors[label], 5, tipLength=0.3)

        # display image
        show_image(out, downsamp=2)

        # handle GUI tasks
        key = cv2.waitKey(props.t_ms)
        if key == ord('q'):
            break

    tock = perf_counter()

    print('Total frames: {}'.format(frames))
    print('Elapsed time: {}'.format(tock-tick))
    print('Throughput: {:0.3f}'.format(frames/(tock-tick)))

if __name__ == '__main__':
    main()