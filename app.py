from src.pipeline.predict_pipeline import PredictionPipeline,CustomData
from flask import Flask,request,jsonify,render_template

app=Flask(__name__)

@app.route("/",methods=['POST','GET'])

def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        
        ##final data input
        final_data=data.get_data_dataframe()
        
        ##prediction of the final data input
        prediction_pipeline=PredictionPipeline()
        pred=prediction_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        return render_template("result.html",final_result=result)
    
if __name__=="__main__":
    app.run(debug=True,port=2020,host='0.0.0.0')        
        