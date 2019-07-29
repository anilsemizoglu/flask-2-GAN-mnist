# following lines are fixes to my weird python environment errors
import matplotlib
matplotlib.use('Agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# above lines are fixes to my weird python environment errors

from flask import Flask, render_template, request, jsonify
from generate_number import generate_num, InputForm

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('graphs.html')


@app.route('/generate_random_number',methods=['GET','POST'])
def generate_random_number():
    digit= 2
    confidence=.5
    graph1_url=-1
    prob=-1
    i=-1
    digit = request.form.get('digit', 0, type=float)
    confidence = request.form.get('confidence', 0, type=float)
    graph1_url, prob, i = generate_num(digit,confidence)
    # print(graph1_url)
    # return render_template('graphs.html',graph1=graph1_url,prob=prob,i=i,digit=int(digit))
    return jsonify(result=graph1_url,prob=str(prob))

if __name__ == '__main__':
    app.debug = True
    app.run(port=5048)
