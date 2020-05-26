from django.shortcuts import render

# Create your views here.
from .models import MyData
from django.http import HttpResponse
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import numpy as np
import pickle

# Create your views here.
def home(request):
    return render(request,'Hm.html')

def display(request):
    return render(request,'application.html')


def save(request):
    if request.method=="POST":
        s=MyData()
        s.Name=request.POST.get('ht_name')
        s.Last_Name=request.POST.get('ht_Lname')
        s.Mobile=request.POST.get('ht_Mnum')
        s.Email=request.POST.get('ht_Email')
        s.Gender=request.POST.get('ht_Male')
        s.TOFEL=request.POST.get('ht_tofel')
        s.GRE=request.POST.get('ht_gre')
        s.UNI_rating=request.POST.get('ht_Uni_rating')
        s.SOP=request.POST.get('ht_sop')
        s.LOR=request.POST.get('ht_lor')
        s.CGPA=request.POST.get('ht_cgpa')
        s.Research_Ex=request.POST.get('ht_research')
        s.save()

        #read the data in data frame
        data=[[s.GRE,s.TOFEL,s.UNI_rating,s.SOP,s.LOR,s.CGPA,s.Research_Ex]]
        newx=pd.DataFrame(data,columns=["GRE_Score","TOEFL_Score","University_Rating","SOP","LOR","CGPA","Research"])

        
        #loading model and data using  pickle
        filename = 'C:\\Users\\Vipin Kumar\\Desktop\\admission\\myapp\\data\\admission_model.sav'

 
        # load the model from disk

        model,x = pickle.load(open(filename, 'rb'))

        #apply minMax scaler for scaling the data
        scalerX = MinMaxScaler(feature_range=(0, 1))
        x[x.columns] = scalerX.fit(x[x.columns])
        
        newx[newx.columns] = scalerX.transform(newx[newx.columns])


        #here we predict the score on new data
        y_predict = model.predict(newx)
        
        #return HttpResponse('Record submitted successfully')
        return render(request,'Output.html', {'score':y_predict})

    
 
     
        
        


