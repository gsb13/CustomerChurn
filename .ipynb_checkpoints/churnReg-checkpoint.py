from flask import Flask, render_template, request,jsonify
import os
import  numpy as np
import pickle

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():

    # tenure	PhoneService	PaperlessBilling	MonthlyCharges	Churn	One year	
    # Two year	Credit card (automatic)	Electronic check	Mailed check


    tenure=int(request.form['tenure'])
    PhoneService=int(request.form['PhoneService'])
    PaperlessBilling=int(request.form['PaperlessBilling'])
    MonthlyCharges = float(request.form['MonthlyCharges'])
    Churn = int(request.form['Churn'])
    Oneyear = int(request.form['Oneyear'])
    Twoyear = int(request.form['Twoyear'])
    Creditcardautomatic = int(request.form['Creditcardautomatic'])
    Electroniccheck = int(request.form['Electroniccheck'])
    Mailedcheck = int(request.form['Mailedcheck'])

    x=np.array([tenure,PhoneService,PaperlessBilling,MonthlyCharges,Churn,Oneyear,Twoyear,
                Creditcardautomatic,Electroniccheck,Mailedcheck]).reshape(1,-1)


    model = pickle.load(open('ML_Model.pkl', 'rb'))

    

    Y_pred=model.predict(x)
    charge =(Y_pred[0])

   
    return jsonify({'Total charge is': charge})
    

if __name__=="__main__":
    app.run(debug=True)