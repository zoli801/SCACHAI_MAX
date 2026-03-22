from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)


@app.route('/')
def i():
    return render_template('index.html')


@app.route('/d')
def g_d():
    with open('db.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    r = []
    for k, v in d.items():
        r.append({'i': k, 'n': v[0], 's': v[1], 'f': v[2]})  # убрать int если что то не работает
    r.sort(key=lambda x: x['s'])
    return jsonify(r)


if __name__ == '__main__':
    app.run(debug=True)
