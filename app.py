from flask import Flask, render_template, request
app = Flask(__name__)


print("Main file")


from recommendor import recommend_patents
import pandas as pd
import numpy as np
@app.route("/")
def index():
    return render_template("profile.html")


@app.route("/getRecommendations", methods=['POST'])
def profile():
       input_user = request.form['input']
       search_by = request.form['searchBy']
       result = recommend_patents(input_user, search_by)
       #result.to_csv(r'result.csv')
       #csv = pd.read_csv("result.csv")
       return render_template("results.html", tables=[result.to_html(classes='data')], titles=result.columns.values)


if __name__ == "__main__":
    app.run()

