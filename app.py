from flask import Flask, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__, static_url_path='/static')

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="s238zXF44rITGe20Axaz"
)

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


if __name__ == '__main__':
    app.run(debug=True)
