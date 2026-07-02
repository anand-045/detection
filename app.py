import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

VISION_KEY = os.environ.get("VISION_KEY", "")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "").rstrip("/")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def _post_v4(image_url, image_bytes, features):
    """Image Analysis 4.0 call. features is a comma separated string,
    e.g. 'Tags,Read,People'."""
    api_url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        f"?api-version=2023-10-01&features={features}"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    if image_url:
        headers["Content-Type"] = "application/json"
        return requests.post(api_url, headers=headers, json={"url": image_url}, timeout=20)
    headers["Content-Type"] = "application/octet-stream"
    return requests.post(api_url, headers=headers, data=image_bytes, timeout=20)


def _post_v32(image_url, image_bytes, path, params):
    """Legacy v3.2 call, used for features not present in Image Analysis 4.0
    (Brands, Landmarks). path is appended after /vision/v3.2/."""
    api_url = f"{VISION_ENDPOINT}/vision/v3.2/{path}"
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    if image_url:
        headers["Content-Type"] = "application/json"
        return requests.post(api_url, headers=headers, params=params, json={"url": image_url}, timeout=20)
    headers["Content-Type"] = "application/octet-stream"
    return requests.post(api_url, headers=headers, params=params, data=image_bytes, timeout=20)


@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({
            "error": "Azure Vision not configured. Set VISION_KEY and "
                     "VISION_ENDPOINT in App Service -> Configuration -> "
                     "Application settings, then restart the app."
        }), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    image_bytes = None
    if image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
    elif not image_url:
        return jsonify({"error": "No image URL or image data provided"}), 400

    # --- Primary call: Image Analysis 4.0 (Tags, Read, People) ---
    try:
        resp = _post_v4(image_url, image_bytes, "Tags,Read,People")
    except requests.RequestException as exc:
        return jsonify({"error": f"Request to Azure AI Vision failed: {exc}"}), 502

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text or "Unexpected response from Azure AI Vision"}

    if not resp.ok:
        return jsonify(body), resp.status_code

    # --- Secondary call: Brands (v3.2 only, best-effort) ---
    try:
        brands_resp = _post_v32(
            image_url, image_bytes, "analyze", {"visualFeatures": "Brands"}
        )
        if brands_resp.ok:
            brands_json = brands_resp.json()
            body["brandsResult"] = {"values": brands_json.get("brands", [])}
        else:
            body["brandsResult"] = {"values": [], "error": brands_resp.text[:300]}
    except requests.RequestException as exc:
        body["brandsResult"] = {"values": [], "error": str(exc)}

    # --- Tertiary call: Landmarks (v3.2 domain model, best-effort) ---
    try:
        landmarks_resp = _post_v32(
            image_url, image_bytes, "models/landmarks/analyze", {"model": "landmarks"}
        )
        if landmarks_resp.ok:
            landmarks_json = landmarks_resp.json()
            values = landmarks_json.get("result", {}).get("landmarks", [])
            body["landmarksResult"] = {"values": values}
        else:
            body["landmarksResult"] = {"values": [], "error": landmarks_resp.text[:300]}
    except requests.RequestException as exc:
        body["landmarksResult"] = {"values": [], "error": str(exc)}

    return jsonify(body), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": bool(VISION_KEY and VISION_ENDPOINT)})


if __name__ == "__main__":
    app.run(debug=True)