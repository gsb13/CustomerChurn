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
    TotalCharges = float(request.form['TotalCharges'])
    Oneyear = int(request.form['Oneyear'])
    Twoyear = int(request.form['Twoyear'])
    Creditcardautomatic = int(request.form['Creditcardautomatic'])
    Electroniccheck = int(request.form['Electroniccheck'])
    Mailedcheck = int(request.form['Mailedcheck'])

    x=np.array([tenure,PhoneService,PaperlessBilling,MonthlyCharges,TotalCharges,Oneyear,Twoyear,
                Creditcardautomatic,Electroniccheck,Mailedcheck]).reshape(1,-1)


    model = pickle.load(open('cls_model.pkl', 'rb'))

    

    Y_pred=model.predict(x)
    churn =(Y_pred[0])

    if churn=='No':
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

    


   
    #return jsonify({'Churn Prediction is ': churn})
    

if __name__=="__main__":
    app.run(debug=True,port=2000)