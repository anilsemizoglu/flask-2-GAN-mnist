# following lines are fixes to my weird python environment errors
import matplotlib
matplotlib.use('Agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# above lines are fixes to my weird python environment errors

from flask import Flask, render_template, request
from generate_number import generate_num, InputForm

app = Flask(__name__)

# @app.route('/')
# def hello_world2():
#
#     return render_template('graphs.html')

@app.route('/',methods=['GET','POST'])
def generate_random_number():
    digit= 1
    confidence=.5
    graph1_url=-1
    prob=-1
    i=-1
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        digit = form.digit.data
        confidence = form.confidence.data
        graph1_url,prob, i = generate_num(form.digit.data,form.confidence.data)
    return render_template('graphs.html',graph1=graph1_url,prob=prob,i=i,digit=int(digit),form=form)

if __name__ == '__main__':
    app.debug = True
    app.run(port=5049)
