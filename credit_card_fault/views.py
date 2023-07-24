from django.shortcuts import render,HttpResponse
from django.views.decorators.csrf import csrf_exempt
from src.pipelines.prediction_pipeline import PredictPipeline,CustomData

@csrf_exempt
def home(request):
    pipe=CustomData()
    columns=pipe.columns
    try:
        if request.method=='POST':
            data=dict()
            for col in columns[:-1]:
                data[col]=[request.POST[col]]
            data[columns[-1]]=[1]
            df=pipe.get_dataframe(data)
            predict=PredictPipeline().predict(df)
            result="Faulty"
            if predict==0:
                result="Not Faulty"
            return HttpResponse(f"<h1>{result}")
    except Exception as e:
        return HttpResponse("<h1>Data was not in the correct format</h1>")
    return render(request,"index.html",{"columns":columns[:-1]})