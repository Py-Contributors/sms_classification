from flask import Flask, request, jsonify
from prediction import predict_txt

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    # get text data from post request
    text = request.form['text']
    output = predict_txt(text)

    return jsonify({'prediction': output})


def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()
