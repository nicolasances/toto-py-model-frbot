from flask import Flask, jsonify, request, Response
import dlg.PredictDlg as predictor

app = Flask(__name__)

@app.route('/')
def smoke():
    return jsonify({
        "api": "model-frbot",
        "status": "running"
    })

@app.route('/predict', methods=['POST'])
def predict():

    try: 
        resp = Response(response=predictor.do(request.json), status=200)
        resp.headers['Content-Type'] = 'application/json'

        return resp

    except KeyError as e: 
        return jsonify({"code": 400, "message": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)