import os
import io
import sys
import cv2
import base64
import platform
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

INPUT_PATH = os.path.join(os.getcwd(), "input")


class Helpers(object):
    @staticmethod
    def decode_image(imageData) -> np.ndarray:
        _, imageData = imageData.split(",")[0], imageData.split(",")[1]
        image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
        return image

    @staticmethod
    def show_image(
        image: np.ndarray, cmap: str = "gnuplot2", title: str = None
    ) -> None:
        plt.figure()
        plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
        plt.axis("off")
        if title:
            plt.title(title)
        if platform.system() == "Windows":
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state("zoomed")
        plt.show()

    @staticmethod
    def show_images(
        image_1: np.ndarray,
        image_2: np.ndarray,
        cmap_1: str = "gnuplot2",
        cmap_2: str = "gnuplot2",
        title_1: str = None,
        title_2: str = None,
    ) -> None:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image_1, cmap=cmap_1)
        plt.axis("off")
        if title_1:
            plt.title(title_1)
        plt.subplot(1, 2, 2)
        plt.imshow(image_2, cmap=cmap_2)
        plt.axis("off")
        if title_2:
            plt.title(title_2)
        if platform.system() == "Windows":
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state("zoomed")
        plt.show()

    @staticmethod
    def draw_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def draw_detections(
        image: np.ndarray, face_detections: tuple, eye_detections: tuple = None
    ):
        if eye_detections is None:
            for x, y, w, h in face_detections:
                cv2.rectangle(
                    img=image,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),
                    thickness=2,
                )
        else:
            for x1, y1, w1, h1 in face_detections:
                for x2, y2, w2, h2 in eye_detections:
                    cv2.rectangle(
                        img=image[y1 : y1 + h1, x1 : x1 + w1],
                        pt1=(x2, y2),
                        pt2=(x2 + w2, y2 + h2),
                        color=(255, 0, 0),
                        thickness=2,
                    )
                cv2.rectangle(
                    img=image,
                    pt1=(x1, y1),
                    pt2=(x1 + w1, y1 + h1),
                    color=(0, 255, 0),
                    thickness=2,
                )


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--base-url", "-bu")
    args_3: tuple = ("--model", "-mo")
    args_4: tuple = ("--filename-1", "-f1")
    args_5: tuple = ("--filename-2", "-f2")

    mode: str = "image"
    base_url: str = "http://localhost:65000"
    model: str = "classify"
    filename_1: str = "Test_1.png"
    filename_2: str = "Test_2.png"

    if args_1[0] in sys.argv:
        mode: str = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_2[0] in sys.argv:
        base_url: str = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_3[0] in sys.argv:
        model: str = sys.argv[sys.argv.index(args_3[0]) + 1]
    if args_4[0] in sys.argv:
        filename_1: str = sys.argv[sys.argv.index(args_4[0]) + 1]
    if args_5[0] in sys.argv:
        filename_2: str = sys.argv[sys.argv.index(args_5[0]) + 1]

    if args_1[1] in sys.argv:
        mode: str = sys.argv[sys.argv.index(args_1[1]) + 1]
    if args_2[1] in sys.argv:
        base_url: str = sys.argv[sys.argv.index(args_2[1]) + 1]
    if args_3[1] in sys.argv:
        model: str = sys.argv[sys.argv.index(args_3[1]) + 1]
    if args_4[1] in sys.argv:
        filename_1: str = sys.argv[sys.argv.index(args_4[1]) + 1]
    if args_5[1] in sys.argv:
        filename_2: str = sys.argv[sys.argv.index(args_5[1]) + 1]

    if mode != "image" and mode != "realtime":
        print(
            "\n"
            + 50 * "*"
            + "\n\n"
            + "Invalid Mode. Only supports 'image' or 'realtime'"
            + "\n\n"
            + 50 * "*"
            + "\n"
        )
        exit()

    if (
        model != "classify"
        and model != "detect"
        and model != "segment"
        and model != "remove"
        and model != "replace"
        and model != "depth"
        and model != "face-detect"
        and model != "face-recognize"
    ):
        print(
            "\n"
            + 50 * "*"
            + "\n\n"
            + f"{model.title()} is an invalid model type"
            + "\n\n"
            + 50 * "*"
            + "\n"
        )
        exit()

    if mode == "image":
        if filename_1 not in os.listdir(INPUT_PATH):
            print(
                "\n"
                + 50 * "*"
                + "\n\n"
                + f"{filename_1} not found in input directory"
                + "\n\n"
                + 50 * "*"
                + "\n"
            )
            exit()

        if model != "replace" and model != "face-recognize":
            files = {"file": open(os.path.join(INPUT_PATH, filename_1), "rb")}

        else:
            if filename_2 not in os.listdir(INPUT_PATH):
                print(
                    "\n"
                    + 50 * "*"
                    + "\n\n"
                    + f"{filename_2} not found in input directory"
                    + "\n\n"
                    + 50 * "*"
                    + "\n"
                )
                exit()

            files = {
                "file1": open(os.path.join(INPUT_PATH, filename_1), "rb"),
                "file2": open(os.path.join(INPUT_PATH, filename_2), "rb"),
            }

        response = requests.request(
            method="POST",
            url=f"{base_url}/{model}",
            files=files,
        )

        if response.status_code == 200:
            if model == "classify":
                print(
                    "\n"
                    + 50 * "*"
                    + "\n\n"
                    + response.json()["label"]
                    + "\n\n"
                    + 50 * "*"
                    + "\n"
                )

            if model == "detect":
                image = cv2.imread(
                    os.path.join(INPUT_PATH, filename_1), cv2.IMREAD_COLOR
                )
                cv2.rectangle(
                    image,
                    (response.json()["box"][0], response.json()["box"][1]),
                    (response.json()["box"][2], response.json()["box"][3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    response.json()["label"],
                    (response.json()["box"][0] - 10, response.json()["box"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                Helpers().show_image(
                    image,
                    title=f"{response.json()['label']} ({response.json()['score']})",
                )

            if model == "segment":
                title: str = ""
                for label in response.json()["labels"]:
                    title += f"{label},"
                title = title[:-1]
                image = Helpers().decode_image(response.json()["imageData"])
                Helpers().show_image(image, title=title)

            if model == "remove":
                image = cv2.imread(
                    os.path.join(INPUT_PATH, filename_1), cv2.IMREAD_COLOR
                )
                Helpers().show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=Helpers().decode_image(response.json()["bglessImageData"]),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="BG Removed Image",
                )

            if model == "replace":
                image = cv2.imread(
                    os.path.join(INPUT_PATH, filename_1), cv2.IMREAD_COLOR
                )
                Helpers().show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=Helpers().decode_image(
                        response.json()["bgreplaceImageData"]
                    ),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="BG Replaced Image",
                )

            if model == "depth":
                image = cv2.imread(
                    os.path.join(INPUT_PATH, filename_1), cv2.IMREAD_COLOR
                )
                Helpers().show_images(
                    image_1=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB),
                    image_2=Helpers().decode_image(response.json()["imageData"]),
                    cmap_1="gnuplot2",
                    cmap_2="gnuplot2",
                    title_1="Original",
                    title_2="Depth Image",
                )

            if model == "face-detect":
                face_detections = response.json()["face_detections"]
                Helpers().draw_detections(image=image, face_detections=face_detections)
                Helpers().show_image(image, title="Face Detections")

            if model == "face-recognize":
                print(f"Simialrity : {float(response.json()['cosine_similarity']):.2f}")

        else:
            print(f"Error {response.status_code} : {response.reason}")

    else:
        raise NotImplementedError("In Process")

        # if platform.system() != "Windows":
        #     cap = cv2.VideoCapture(0)
        # else:
        #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        # while True:
        #     ret, frame = cap.read()
        #     if not ret: break
        #     frameData = encode_image_to_base64(image=frame)
        #     payload = {
        #         "imageData" : frameData
        #     }

        #     response = requests.request(method="POST", url=f"{base_url}/{model}", json=payload)

        #     if response.status_code == 200 and response.json()["statusCode"] == 200:
        #         if model == "classify":
        #             cv2.putText(frame, response.json()["label"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #         if model == "detect":
        #             cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
        #             cv2.putText(frame, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #         if model == "segment":
        #             disp_frameData = response.json()["imageData"]
        #             frame = decode_image(disp_frameData)

        #         if model == "face":
        #             face_detections = response.json()["face_detections"]
        #             draw_detections(image=frame, face_detections=face_detections)

        #     else:
        #         cv2.putText(frame, "Error", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #     cv2.imshow("Processed", frame)

        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        # cap.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)
