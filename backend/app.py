from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy import ndimage
import base64
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = tf.keras.models.load_model("model/handwritten.h5")

def center_image(img):
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    shifted_img = ndimage.shift(img, shift=[shifty, shiftx])
    return shifted_img

def preprocess_image(image):
    img = ImageOps.grayscale(image)
    img_arr = np.array(img)

    if np.mean(img_arr) > 127:
        img_arr = 255 - img_arr
    img_arr = cv2.GaussianBlur(img_arr, (3, 3), 0)
    _, binary_img = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return None 

    x, y, w, h = cv2.boundingRect(coords)

    margin = 5
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, img_arr.shape[1])
    y2 = min(y + h + margin, img_arr.shape[0])

    cropped = binary_img[y1:y2, x1:x2]

    h_c, w_c = cropped.shape
    if h_c > w_c:
        new_h = 20
        new_w = int(w_c * (20 / h_c))
    else:
        new_w = 20
        new_h = int(h_c * (20 / w_c))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    final_img = center_image(final_img)
    final_img = final_img.astype(np.float32) / 255.0

    return final_img

def compute_quality_score(img):
    """Returns smooth quality score between 0 and 1."""
    ink_ratio = np.sum(img > 0.15) / img.size

    quality = 4 * ink_ratio * (1 - ink_ratio)

    # Slightly increase confidence
    quality = quality * 2.2

    quality = max(0.1, min(quality, 1.0))

    return quality

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image data"}), 400

    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    processed = preprocess_image(image)

    if processed is None:
        return jsonify({
            "digit": None,
            "confidence": 0,
            "error": "No digit detected (blank image)"
        })

    input_img = processed.reshape(1, 28, 28, 1).astype(np.float32)

    pred = model.predict(input_img, verbose=0)
    digit = int(np.argmax(pred[0]))
    model_confidence = float(np.max(pred[0]))

    quality_score = compute_quality_score(processed)

    final_confidence = model_confidence * quality_score

    return jsonify({
        "digit": digit,
        "model_confidence": model_confidence,
        "quality_score": quality_score,
        "confidence": final_confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
