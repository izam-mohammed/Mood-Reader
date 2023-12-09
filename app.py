from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from sentimentAnalysis.pipeline.prediction import PredictionPipeline
from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.utils.common import load_json
from pathlib import Path


app = Flask(__name__)  # initializing a flask app


@app.route("/", methods=["GET"])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route("/train", methods=["GET"])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route(
    "/predict", methods=["POST", "GET"]
)  # route to show the predictions in a web UI
def index():
    if request.method == "POST":
        try:
            #  reading the inputs given by the user
            text = request.form["text"]
            config = ConfigurationManager()
            config = config.get_prediction_config()
            data_path = config.data_path
            with open(data_path, "w") as f:
                f.write(text)

            obj = PredictionPipeline()
            obj.main()

            predition_path = config.prediction_file
            file = load_json(path=Path(predition_path))
            predict = file["prediction"]

            sentiment_color = (
                "background-color: red;" if predict == 0 else "background-color: green;"
            )
            sentiment = "Negative" if not predict else "Positive"
            return render_template(
                "result.html",
                sentiment=sentiment,
                sentiment_color="black",
                bg_style=sentiment_color,
            )

        except Exception as e:
            print("The Exception message is: ", e)
            return render_template("404.html")

    else:
        return render_template("index.html")


@app.route("/predict_data", methods=["POST"])
def pred():
    try:
        text = request.get_json().get("text")
        config = ConfigurationManager()
        config = config.get_prediction_config()
        data_path = config.data_path
        with open(data_path, "w") as f:
            f.write(text)

        obj = PredictionPipeline()
        obj.predict()

        predition_path = config.prediction_file
        file = load_json(path=Path(predition_path))
        predict = file["prediction"]

        return jsonify({"result": predict})

    except Exception as e:
        print("The Exception message is: ", e)
        return jsonify({"result": "error"})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port=8080)
