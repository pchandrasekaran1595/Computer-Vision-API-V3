import re
import sys

from sanic import Sanic
from sanic.request import Request
from sanic.exceptions import SanicException
from sanic.response import JSONResponse
from static.utils import models, Helpers


STATIC_PATH: str = "static"
VERSION: str = "1.0.0"
PORT: int = 4040

app = Sanic("Computer-Vision-API-V3")
app.static("static", "static")


@app.get("/")
async def get_root(request: Request):
    return JSONResponse(
        body={"statusText": "Root Endpoint of CV-API-V3"},
        status=200,
    )


@app.get("/version")
async def get_version(request: Request):
    return JSONResponse(
        body={"statusText": "Version Fetch Successful", "version": VERSION},
        status=200,
    )


@app.get("/<infer_type:str>")
async def get_infer(request: Request, infer_type: str):
    if (
        not re.match(r"^classify$", infer_type, re.IGNORECASE)
        and not re.match(r"^detect$", infer_type, re.IGNORECASE)
        and not re.match(r"^segment$", infer_type, re.IGNORECASE)
        and not re.match(r"^remove$", infer_type, re.IGNORECASE)
        and not re.match(r"^replace$", infer_type, re.IGNORECASE)
        and not re.match(r"^depth$", infer_type, re.IGNORECASE)
        and not re.match(r"^face-detect$", infer_type, re.IGNORECASE)
        and not re.match(r"^face-recognize$", infer_type, re.IGNORECASE)
    ):
        raise SanicException(
            message=f"{infer_type.title()} is Invalid", status_code=404
        )

    return JSONResponse(
        body={"statusText": f"{infer_type} Endpoint"},
        status=200,
    )


@app.post("/<infer_type:str>")
async def post_infer(request: Request, infer_type: str):
    if (
        not re.match(r"^classify$", infer_type, re.IGNORECASE)
        and not re.match(r"^detect$", infer_type, re.IGNORECASE)
        and not re.match(r"^segment$", infer_type, re.IGNORECASE)
        and not re.match(r"^remove$", infer_type, re.IGNORECASE)
        and not re.match(r"^replace$", infer_type, re.IGNORECASE)
        and not re.match(r"^depth$", infer_type, re.IGNORECASE)
        and not re.match(r"^face-detect$", infer_type, re.IGNORECASE)
        and not re.match(r"^face-recognize$", infer_type, re.IGNORECASE)
    ):
        raise SanicException(
            message=f"{infer_type.title()} is Invalid", status_code=404
        )

    if re.match(r"^classify$", infer_type, re.IGNORECASE):
        image = Helpers().decode_image(request.files.get("file").body)
        label = models[0].infer(image=image)

        return JSONResponse(
            body={
                "statusText": "Classification Inference Complete",
                "label": label,
            },
            status=200,
        )

    elif re.match(r"^detect$", infer_type, re.IGNORECASE):
        image = Helpers().decode_image(request.files.get("file").body)

        label, score, box = models[1].infer(image)

        if label is None:
            return JSONResponse(
                body={
                    "statusText": "No Detections",
                },
                status=406,
            )

        return JSONResponse(
            body={
                "statusText": "Detection Inference Complete",
                "label": label,
                "score": str(score),
                "box": box,
            },
            status=200,
        )

    elif re.match(r"^segment$", infer_type, re.IGNORECASE):
        image = Helpers().decode_image(request.files.get("file").body)

        segmented_image, labels = models[2].infer(image)

        return JSONResponse(
            body={
                "statusText": "Segmentation Inference Complete",
                "labels": labels,
                "imageData": Helpers().encode_image_to_base64(image=segmented_image),
            },
            status=200,
        )

    elif re.match(r"^remove$", infer_type, re.IGNORECASE):
        image = Helpers().decode_image(request.files.get("file").body)

        mask = models[3].infer(image=image)
        for i in range(3):
            image[:, :, i] = image[:, :, i] & mask

        return JSONResponse(
            body={
                "statusText": "Background Removal Complete",
                "maskImageData": Helpers().encode_image_to_base64(image=mask),
                "bglessImageData": Helpers().encode_image_to_base64(image=image),
            },
            status=200,
        )

    elif re.match(r"^replace$", infer_type, re.IGNORECASE):
        image_1 = Helpers().decode_image(request.files.get("file1").body)
        image_2 = Helpers().decode_image(request.files.get("file2").body)

        mask = models[3].infer(image=image_1)
        mh, mw = mask.shape
        image_2 = Helpers().preprocess_replace_bg_image(image_2, mw, mh)
        for i in range(3):
            image_1[:, :, i] = image_1[:, :, i] & mask
            image_2[:, :, i] = image_2[:, :, i] & (255 - mask)

        image_2 += image_1

        return JSONResponse(
            body={
                "statusText": "Background Replacement Complete",
                "bgreplaceImageData": Helpers().encode_image_to_base64(image=image_2),
            },
            status=200,
        )

    elif re.match(r"^depth$", infer_type, re.IGNORECASE):
        image = Helpers().decode_image(request.files.get("file").body)

        image = models[4].infer(image=image)

        return JSONResponse(
            body={
                "statusText": "Depth Inference Complete",
                "imageData": Helpers().encode_image_to_base64(image=image),
            },
            status=200,
        )

    elif re.match(r"^face-detect$", infer_type, re.IGNORECASE):
        image = Helpers().decode_image(request.files.get("file").body)

        face_detections_np = models[5].infer(image)

        if len(face_detections_np) == 0:
            return JSONResponse(
                body={
                    "statusText": "No Detections",
                },
                status=406,
            )

        face_detections: list = []
        for x, y, w, h in face_detections_np:
            face_detections.append([int(x), int(y), int(w), int(h)])

        return JSONResponse(
            body={
                "statusText": "Face Detection Complete",
                "face_detections": face_detections,
            },
            status=200,
        )

    elif re.match(r"^face-recognize$", infer_type, re.IGNORECASE):
        image_1 = Helpers().decode_image(request.files.get("file1").body)
        image_2 = Helpers().decode_image(request.files.get("file2").body)

        cs = models[6].get_cosine_similarity(image_1, image_2)

        if cs is None:
            return JSONResponse(
                body={
                    "statusText": "Possible error in APIData; cannot calculate similarity",
                },
                status=406,
            )

        return JSONResponse(
            body={
                "statusText": "Face Recognition Inference Complete",
                "cosine_similarity": str(cs),
            },
            status=200,
        )


if __name__ == "__main__":
    args_1: tuple = ("-m", "--mode")
    args_2: tuple = ("-w", "--workers")

    mode: str = "local-machine"
    workers: int = 1

    if args_1[0] in sys.argv:
        mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv:
        mode = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv:
        workers = int(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv:
        workers = int(sys.argv[sys.argv.index(args_2[1]) + 1])

    if mode == "local-machine":
        app.run(host="localhost", port=PORT, dev=True, workers=workers)

    elif mode == "local":
        app.run(host="0.0.0.0", port=PORT, dev=True, workers=workers)

    elif mode == "render":
        app.run(host="0.0.0.0", port=PORT, single_process=True, access_log=True)

    elif mode == "prod":
        app.run(host="0.0.0.0", port=PORT, dev=False, workers=workers, access_log=True)

    else:
        print(
            "\n"
            + 50 * "*"
            + "\n\n"
            + "Invalid Run Mode. Only supports (local-machine, local, render, prod)"
            + "\n\n"
            + 50 * "*"
            + "\n"
        )
        exit()
