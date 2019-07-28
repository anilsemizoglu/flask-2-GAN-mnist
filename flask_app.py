from flask import Flask, render_template
from generate_number import generate_num
app = Flask(__name__)

@app.route('/')
def hello_world():
    x1 = [0, 1, 2, 3, 4]
    y1 = [10, 30, 40, 5, 50]
    # graph1_url = build_graph(x1,y1);
    graph1_url,prob, i = generate_num();
    print(i)
    return render_template('graphs.html',graph1=graph1_url,prob=prob,i=i)

if __name__ == '__main__':
    app.debug = True
    app.run(port=5049)
