import cv2
import numpy as np
from argparse import ArgumentParser
from time import perf_counter

from cs229.display import open_window, show_image
from cs229.files import read_video, write_video
from cs229.image import img_to_mask, bound_point
from cs229.contour import find_contours
from cs229.patch import crop_to_contour
from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor
from cs229.train_orientation import PosePredictor
from cs229.load_data_wing import male_fly_patch
from cs229.train_wing import WingPredictor
from cs229.timing import Profiler

def arrow_from_point(img, point, length, angle, color):
    ax = point[0] + length * np.cos(angle)
    ay = point[1] - length * np.sin(angle)
    tip = bound_point((ax, ay), img)
    cv2.arrowedLine(img, point, tip, color, 5, tipLength=0.3)

def main():
    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--write_video', action='store_true')
    parser.add_argument('-i', '--input', type=str, default='test4.mp4')
    parser.add_argument('-o', '--output', type=str, default='output.avi')
    args = parser.parse_args()

    # prepare video
    cap, props = read_video(args.input)
    _, img = cap.read()
    img = img[:, :, 0]
    mask = img_to_mask(img)

    if args.write_video:
        video_writer = write_video(args.output, props)

    # load predictors
    is_fly_predictor = IsFlyPredictor()
    id_predictor = IdPredictor()
    pose_predictor = {type: PosePredictor(type) for type in ['male', 'female']}
    wing_predictor = WingPredictor()

    # display-specific actions
    if not args.no_display:
        open_window()
        colors = {'female': (255, 0, 0), 'male': (0, 0, 255), 'both': (255, 0, 255), 'neither': None}

    frames = 0
    tick = perf_counter()
    prof = Profiler()

    while True:
        frame_start = perf_counter()

        # read frame
        prof.tick('I/O')
        ok, img = cap.read()
        prof.tock('I/O')

        if not ok:
            break

        if not args.no_display:
            out = img.copy()

        img = img[:, :, 0]

        frames += 1

        # extract contours
        prof.tick('Find fly contours')
        contours = find_contours(img, mask=mask, type='core')

        # draw contours with a color associated with the class
        contours_by_label = {'neither': [], 'one': [], 'both': []}

        # sort contours into bins: zero flies, one fly and two flies
        for contour in contours:
            label = is_fly_predictor.predict(contour)
            contours_by_label[label].append(contour)

        prof.tock('Find fly contours')

        results = {}
        if len(contours_by_label['one'])==2 and len(contours_by_label['both'])==0:
            prof.tick('ID as male/female')

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

            prof.tock('ID as male/female')

            prof.tick('Determine orientation')

            for type in ['male', 'female']:
                result = results[type]
                (cx, cy), angle = pose_predictor[type].predict(result['patch'])
                result.update(dict(cx=cx, cy=cy, angle=angle))

            prof.tock('Determine orientation')

            # predict wing angle
            prof.tick('Determine wing angles')
            patch_m = male_fly_patch(img, mask, (results['male']['cx'], results['male']['cy']),
                                     results['male']['angle'])

            if patch_m is not None:
                wing_angle_right, wing_angle_left = wing_predictor.predict(patch_m)
                if wing_angle_right is not None:
                    results['male']['wing_angle_right'] = wing_angle_right
                if wing_angle_left is not None:
                    results['male']['wing_angle_left'] = wing_angle_left

            prof.tock('Determine wing angles')
        elif len(contours_by_label['one'])==0 and len(contours_by_label['both'])==1:
            results['both'] = dict(contour=contours_by_label['both'][0])
            results['both']['patch'] = crop_to_contour(img, results['both']['contour'])
            results['both']['cx'], results['both']['cy'] = results['both']['patch'].estimate_center(absolute=True)
        else:
            print('Unexpected case.')
            continue

        # when profiling, skip the drawing steps
        if args.no_display:
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
            arrow_from_point(out, center, 0.3*MA, result['angle'], colors[label])

            if label == 'male':
                if 'wing_angle_right' in result:
                    arrow_angle = result['angle'] + result['wing_angle_right'] - np.pi
                    arrow_from_point(out, center, 0.3*MA, arrow_angle, (255, 255, 0))
                if 'wing_angle_left' in result:
                    arrow_angle = result['angle'] - result['wing_angle_left'] - np.pi
                    arrow_from_point(out, center, 0.3*MA, arrow_angle, (255, 255, 0))

        # display image
        show_image(out, downsamp=2)

        if args.write_video:
            video_writer.write(out)

        # figure out how much extra time the GUI should wait before proceeding
        t_frame_ms = 1e3*(perf_counter() - frame_start)
        t_extra_ms = int(round(props.t_ms - t_frame_ms))

        # handle GUI tasks
        key = cv2.waitKey(max(t_extra_ms, 1))

        # process input keys
        if key == ord('q'):
            break

    tock = perf_counter()

    prof.stop()

    if args.write_video:
        video_writer.release()

    print('Total frames: {}'.format(frames))
    print('Elapsed time: {}'.format(tock-tick))
    print('Throughput: {:0.3f}'.format(frames/(tock-tick)))

if __name__ == '__main__':
    main()