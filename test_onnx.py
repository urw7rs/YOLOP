import os
import socket
import cv2
import argparse
import onnxruntime as ort
import numpy as np
from sklearn import cluster

UDP_IP = os.environ["UDP_IP"]
UDP_PORT = int(os.environ["UDP_PORT"])
HEADER = "$CAM_LANE"


def resize_unscale(img, new_shape=(320, 320), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)

    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh : dh + new_unpad_h, dw : dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def load_yolop(onnx_path="yolop-320-320.onnx"):
    ort.set_default_logger_severity(4)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    ort_session = ort.InferenceSession(
        onnx_path,
        sess_option=so,
        providers=providers,
    )

    print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    return ort_session


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0")
parser.add_argument("--flip", action="store_true")

parser.add_argument("--onnx_path", type=str, default=None)
parser.add_argument("--img_size", type=int, default=320)

parser.add_argument("--close", action="store_true")
parser.add_argument("--kernel_size", type=int, default=7)

parser.add_argument("--top", type=float, default=(480 - 155) / 480)
parser.add_argument("--middle", type=float, default=(480 - 135) / 480)
parser.add_argument("--bottom", type=float, default=(480 - 115) / 480)

parser.add_argument("--debug", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    sess_options = ort.SessionOptions()

    if args.source.isnumeric():
        args.source = int(args.source)

    cap = cv2.VideoCapture(args.source)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"width: {width} height: {height}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    R_TOP = (480 - 155) / 480
    R_MIDDLE = (480 - 135) / 480
    R_BOTTOM = (480 - 115) / 480

    if args.onnx_path is None:
        args.onnx_path = f"weights/yolop-{args.img_size}-{args.img_size}.onnx"
    ort_session = load_yolop(onnx_path=args.onnx_path)

    dbscan = cluster.DBSCAN(eps=1, min_samples=5, n_jobs=-1)

    if args.close:
        kernel_size = args.kernel_size
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size)
        )

    # for coloring clusters
    colors = [[0, np.random.randint(255), np.random.randint(255)] for _ in range(100)]

    frame_counter = 0

    while True:
        ret, img_bgr = cap.read()

        if not ret:
            print("failed to grab img_bgr")
            break

        if args.flip:
            img_bgr = cv2.flip(img_bgr, 0)
            img_bgr = cv2.flip(img_bgr, 1)

        frame_counter += 1

        height, width, _ = img_bgr.shape

        # convert to RGB
        img_rgb = img_bgr[:, :, ::-1].copy()

        # resize & normalize
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(
            img_rgb, (args.img_size, args.img_size)
        )

        TOP = int(new_unpad_h * args.top)
        MIDDLE = int(new_unpad_h * args.middle)
        BOTTOM = int(new_unpad_h * args.bottom)

        img = canvas.copy().astype(np.float32)  # (3,320,320) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,320,320)

        # inference: (1,n,6) (1,2,320,320) (1,2,320,320)
        det_out, da_seg_out, ll_seg_out = ort_session.run(
            ["det_out", "drive_area_seg", "lane_line_seg"],
            input_feed={"images": img},
        )

        # select ll segment area.
        ll_seg_out = ll_seg_out[:, :, dh : dh + new_unpad_h, dw : dw + new_unpad_w]

        (ll_seg_mask,) = np.argmax(ll_seg_out, axis=1)  # (?,?) (0|1)

        if args.close:
            ll_seg_mask = cv2.morphologyEx(
                ll_seg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1
            )

        masked_ll_seg_mask = ll_seg_mask.copy()

        # remove top portion where lanes can merge
        masked_ll_seg_mask[:TOP, :] = 0
        masked_ll_seg_mask[BOTTOM + 1 :, :] = 0

        # get distance from center to lane centers
        points_of_interest = np.zeros((2, 3, 2)).astype(int)
        points_of_interest[:, :, 0] = [TOP, MIDDLE, BOTTOM]
        points_of_interest[:, :, 1].fill(new_unpad_w // 2)

        if np.any(masked_ll_seg_mask == 1):
            y, x = np.where(masked_ll_seg_mask == 1)
            lane_coords = np.stack([y, x]).T

            # label coordinates using DBSCAN
            dbscan.fit(lane_coords)

            lanes = []
            distances = []
            found_poi = []

            labels = dbscan.labels_
            # exclude outlier with label -1
            for label in range(labels.max() + 1):
                # extract coordinates
                (index,) = np.where(labels == label)
                lane_y = y[index]
                lane_x = x[index]

                # average along x axis
                lane = []
                for unique_y in np.unique(lane_y):
                    (y_index,) = np.where(lane_y == unique_y)
                    mean_x = int(lane_x[y_index].mean())
                    lane.append(np.array([unique_y, mean_x]))
                lane = np.stack(lane)  # [y; x]
                lanes.append(lane)

                # extract points of interest
                points = np.zeros((3, 2)).astype(int)
                points[:, 0] = [TOP, MIDDLE, BOTTOM]
                # points[:, 1].fill(new_unpad_w // 2)
                for i, target_y in enumerate([TOP, MIDDLE, BOTTOM]):
                    target_x = np.extract(
                        lane[:, 0] == target_y,
                        lane[:, 1],
                    )

                    if len(target_x):
                        points[i, 1] = target_x

                found_poi.append(points)

                # distance between center and lane middle point
                distance = np.abs(points[1, 1] - new_unpad_w / 2)
                distances.append(distance)

            # reorder lanes
            index = np.argsort(distances)[:2]
            for i, j in enumerate(index.tolist()):
                points_of_interest[i] = found_poi[j]

        # transform coordinates
        tf_poi = points_of_interest.copy()
        tf_poi[:, :, 1] = tf_poi[:, :, 1] - new_unpad_w // 2
        tf_poi[:, :, 0] = new_unpad_h - tf_poi[:, :, 0]

        # send message
        message = [HEADER]
        for points in tf_poi.tolist():
            for y, x in points:
                message.append(f"{x} {y}")
        message = ", ".join(message)
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        if args.debug is False:
            continue

        height, width, _ = img_rgb.shape

        cluster_seg = np.zeros((new_unpad_h, new_unpad_w, 3))
        cluster_seg[ll_seg_mask == 1] = (255, 0, 0)
        if np.any(masked_ll_seg_mask == 1):
            y, x = np.where(masked_ll_seg_mask == 1)
            for label in labels:
                (index,) = np.where(labels == label)
                cluster_seg[y[index], x[index]] = colors[label]

        cluster_labels = cluster_seg.astype(np.uint8)
        cluster_labels = cv2.resize(
            cluster_labels, (width, height), interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow("cluster_labels", cluster_labels)

        # convert to BGR
        cluster_seg = cluster_seg[..., ::-1]
        cluster_mask = np.mean(cluster_seg, 2)
        img_merge = canvas[dh : dh + new_unpad_h, dw : dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        # merge: resize to original size
        img_merge[cluster_mask != 0] = (
            img_merge[cluster_mask != 0] * 0.5 + cluster_seg[cluster_mask != 0] * 0.5
        )
        img_merge = img_merge.astype(np.uint8)

        img_merge = cv2.resize(
            img_merge, (width, height), interpolation=cv2.INTER_LINEAR
        )

        scale = width // new_unpad_w
        # draw points
        for points in points_of_interest.tolist():
            for y, x in points:
                cv2.circle(
                    img_merge,
                    (scale * x, scale * y),
                    5,
                    (255, 0, 0),
                    5,
                )

        # draw lines
        for y in [TOP, MIDDLE, BOTTOM]:
            cv2.line(img_merge, (0, scale * y), (width, scale * y), (0, 0, 255), 2)

        cv2.line(
            img_merge,
            (width // 2, 0),
            (width // 2, height),
            (0, 0, 255),
            2,
        )

        # ll: resize to original size
        ll_seg_mask = ll_seg_mask * 255
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = cv2.resize(
            ll_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow("img_merge", img_merge)
        cv2.imshow("ll_seg_mask", ll_seg_mask)

        k = cv2.waitKey(1)
        if k % 256 == ord("q"):
            # q pressed
            print("q pressed, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            for points in points_of_interest.tolist():
                for y, x in points:
                    print(f"x: {x * 2}, y: {(new_unpad_h - y) * 2}")
            img_name = f"opencv_frame_{frame_counter}.png"
            cv2.imwrite(img_name, img_merge)
            print(f"{img_name} written!")

    cap.release()

    cv2.destroyAllWindows()
