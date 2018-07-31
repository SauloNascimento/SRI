from flask import Flask, request, render_template, jsonify
from app import app, model

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def my_form_post():
    idNew = request.form['text']
    new = model.get_new(int(idNew))
    neighbors = model.get_5_neighbors(int(idNew))
    return render_template('results.html', new=new, neighbors=neighbors)
