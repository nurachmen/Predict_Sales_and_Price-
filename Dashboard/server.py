from flask import Flask, request, render_template , jsonify
import requests
import numpy as np
import joblib as jb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_url_path='') 

#home route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
      input = request.form

      Price = int(request.form['Price'])
      UsesAdBoosts = int(request.form['UsesAdBoosts'])
      RetailPrice = int(request.form['RetailPrice'])
      AmountRating = int(request.form['AmountRating'])
      Rating = int(request.form['Rating'])
      BadgeLocalProduct = int(request.form['BadgeLocalProduct'])
      BadgeProductQuality = int(request.form['BadgeProductQuality'])
      BadgeFastShipping = int(request.form['BadgeFastShipping'])
      AmountMerchantRating = int(request.form['AmountMerchantRating'])
      MerchantRating = float(request.form['MerchantRating'])
      
      prediksi = ([[Price,UsesAdBoosts,RetailPrice,AmountRating,Rating,BadgeLocalProduct,
                               BadgeProductQuality,BadgeFastShipping,AmountMerchantRating,
                               MerchantRating]])[0]
      
      prediksi_fix = np.array(prediksi).reshape((1,-1))

  
      prediksi_fix1 =  model.predict(prediksi_fix)
      # prediksi = round(prediksi, 0)
      # prediksi_fix = round(prediksi_fix, 0)

      dataHasil = {
        'Price' : Price, 
        'UsesAdBoosts' : UsesAdBoosts,
        'RetailPrice' : RetailPrice,
        'AmountRating' : AmountRating, 
        'Rating' : Rating, 
        'BadgeLocalProduct' : BadgeLocalProduct,
        'BadgeProductQuality' : BadgeProductQuality,
        'BadgeFastShipping' : BadgeFastShipping,
        'AmountMerchantRating' : AmountMerchantRating,
        'MerchantRating' : MerchantRating,
        'predict' : prediksi_fix1
        # 'predict' : prediksi
        
        } 
        

      return render_template('result.html', Hasil=dataHasil)

@app.route('/Visualization')
def displot():
    return render_template('Visualization.html')


if __name__=='__main__':
    model = jb.load('modeljoblib')
    app.run(debug=True)