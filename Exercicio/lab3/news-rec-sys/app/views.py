from flask import Flask, request, render_template, jsonify
from app import app, model

@app.route('/')
def index():
    valdocs = model.getValidationDocs()
    return render_template("index.html", valdocs=valdocs)

@app.route('/', methods=['POST'])
def my_form_post():
    idNew = request.form['selId']
    new = model.get_new(int(idNew))
    neighbors = model.get_5_neighbors(int(idNew))
    return render_template('results.html', new=new, neighbors=neighbors)
