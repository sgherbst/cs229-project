import cv2
import numpy as np
from time import perf_counter

from cs229.display import open_window, show_image
from cs229.files import open_video
from cs229.image import img_to_mask, bound_point
from cs229.contour import find_contours, in_contour
from cs229.patch import crop_to_contour, mask_from_contour
from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor
from cs229.train_orientation import PosePredictor

def main(profile=False):
    # prepare video
    cap, props = open_video('cropped')
    _, img = cap.read()
    img = img[:, :, 0]
    mask = img_to_mask(img)

    # load predictors
    is_fly_predictor = IsFlyPredictor()
    id_predictor = IdPredictor()
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

        if not profile:
            out = img.copy()

        img = img[:, :, 0]

        frames += 1

        # extract contours
        contours = find_contours(img, mask=mask)

        # draw contours with a color associated with the class
        contours_by_label = {'neither': [], 'one': [], 'both': []}

        # sort contours into bins: zero flies, one fly and two flies
        for contour in contours:
            label = is_fly_predictor.predict(contour)
            contours_by_label[label].append(contour)

        results = {}
        if len(contours_by_label['one'])==2 and len(contours_by_label['both'])==0:
            contour_1 = contours_by_label['one'][0]
            contour_2 = contours_by_label['one'][1]

            patch_1 = crop_to_contour(img, contour_1)
            patch_2 = crop_to_contour(img, contour_2)

            label = id_predictor.predict(contour_1, contour_2, patch_1, patch_2)

            if label == 'mf':
                results['male'] = dict(contour=contour_1, patch=patch_1)
                results['female'] = dict(contour=contour_2, patch=patch_2)
            elif label == 'fm':
                results['female'] = dict(contour=contour_1, patch=patch_1)
                results['male'] = dict(contour=contour_2, patch=patch_2)

            for type in ['male', 'female']:
                result = results[type]
                (cx, cy), angle = pose_predictor[type].predict(result['patch'])
                result.update(dict(cx=cx, cy=cy, angle=angle))
        elif len(contours_by_label['one'])==0 and len(contours_by_label['both'])==1:
            results['both'] = dict(contour=contours_by_label['both'][0])
            results['both']['patch'] = crop_to_contour(img, results['both']['contour'])
            results['both']['cx'], results['both']['cy'] = results['both']['patch'].estimate_center(absolute=True)
        else:
            print('Unexpected case.')
            continue

        # when profiling, skip the drawing steps
        if profile:
            continue

        # illustrate the results
        for label, result in results.items():
            # draw outline
            cv2.drawContours(out, [result['contour']], -1, colors[label], 3)

            if label not in ['male', 'female']:
                continue

            # draw center
            center = bound_point((result['cx'], result['cy']), out)
            cv2.circle(out, center, 5, colors[label], -1)

            # draw arrow in direction of orientation
            MA, ma = result['patch'].estimate_axes()
            ax = result['cx'] + 0.3*MA*np.cos(result['angle'])
            ay = result['cy'] - 0.3*MA*np.sin(result['angle'])
            tip = bound_point((ax, ay), out)
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