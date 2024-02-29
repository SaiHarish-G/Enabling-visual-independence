from flask import Flask, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient
import easyocr
from PIL import Image
import io

app = Flask(__name__, static_url_path='/static')

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="s238zXF44rITGe20Axaz"
)

# Initialize easyocr reader
reader = easyocr.Reader(['en'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    # Get the image file from the request
    image_file = request.files['image']

    # Save the image locally
    image_path = 'static/test_currency.jpg'
    image_file.save(image_path)

    # Perform inference on the image
    result = CLIENT.infer(image_path, model_id="indian-currency-detection-elfyf/1")

    print("Inference Result:", result)  # Print out the result object for debugging

    # Return the inference result as JSON
    return jsonify(result)

@app.route('/perform_ocr', methods=['POST'])
def perform_ocr():
    # Get the image file from the request
    image_file = request.files['image']

    # Save the image locally
    image_path = 'static/uploaded_image.jpg'
    image_file.save(image_path)

    # Open the image file
    img = Image.open(image_path)

    # Convert PIL.Image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Perform OCR on the image
    result = reader.readtext(img_byte_arr)

    # Extract text from the OCR result
    extracted_text = ' '.join([result[i][1] for i in range(len(result))])

    # Return the extracted text as JSON
    return jsonify({'extracted_text': extracted_text})

if __name__ == '__main__':
    app.run(debug=True)
