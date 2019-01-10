import datetime
import pickle
import string
from os.path import join
import os
from flask import Flask, render_template, session, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from sqlalchemy import Column, Integer, String, DateTime
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired

import PreProcessing_valid



app = Flask(__name__)
app.config['SECRET_KEY'] = 'abc'

bootstrap = Bootstrap(app)
moment = Moment(app)


class NameForm(FlaskForm):
    name = TextAreaField('Mời Bạn Nhập:', validators=[DataRequired()],render_kw={"rows": 8, "cols": 10})
    submit = SubmitField('Trích Xuất')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    form = NameForm()
    # submit = SubmitField('Trích Xuất')
    dict_new = {}
    if form.validate_on_submit():
        # old_name = form.name.data
        # print(old_name)
        # if old_name is not None:
        #     flash(old_name)

        # if old_name is not None and old_name != form.name.data:
        #     flash('Do you thing there is a human sitting behind the screen!')
        # session['name'] = PreProcessing_valid.PreProcessing(form.name.data)
        session['name'] = form.name.data
        sentences = session['name'].split(".")
        sentences_new = []

        for i in sentences:
            if (i != ''):
                i = i.strip(' ')
                i = PreProcessing_valid.PreProcessing(i)
                sentences_new.append(i)
        print(sentences_new)
        predict = []
        list_file = os.listdir("models_SVC_new")


        for i in sentences_new:
            print(i)
            prediction = []
            pre = []
            text_new = ""
            pre.append(i)
            for model in list_file:
                load_file = open(join("models_SVC_new", model), 'rb')
                clf = pickle.load(load_file)
                t = clf.predict(pre)
                prediction.append(t)

            for j in prediction:
                if (j != "None\n"):
                    j = str(j).strip("\[]'n,")
                    text_new = text_new + j + ", "


            text_new = text_new.strip(', ')
            text_new = text_new+"."

            if (text_new == "."):
                text_new = "None. "
            # print(text_new[len(text_new)-2])
            # text_new = text_new.replace(text_new[len(text_new)-2],".")
            print(text_new)
            dict_new[i] = text_new

        # return redirect(url_for('index'),dict_new)

    dict = {'phy': 50, 'che': 60, 'maths': 70}
    print(dict_new)


    return render_template('index.html', form=form, name="", result = dict_new)

if __name__ == "__main__":
    app.run(host="localhost", port=9000, debug=True)
