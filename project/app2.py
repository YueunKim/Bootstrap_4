from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('regression2.html', menu=menu)
    else:
        slen = float(request.form['slen'])
        swid = float(request.form['swid'])
        plen = float(request.form['plen'])
        pwid = float(request.form['pwid'])
        species = 1.18650 -0.11191 * slen -0.04008 * swid + 0.22865 * plen + 0.60925 * pwid
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':round(species, 4)}
        return render_template('reg_result2.html', menu=menu, iris=iris)

@app.route('/sentiment')
def sentiment():
    pass

@app.route('/classification')
def classification():
    pass

@app.route('/clustering')
def clustering():
    pass

if __name__ == '__main__':
    app.run(debug=True) ## deburg = true : 개발모드
