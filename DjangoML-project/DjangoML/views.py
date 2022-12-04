from django.http import HttpResponse
from django.shortcuts import render
import joblib
def home(request):
    return render(request,"home.html")

def result(request):

    cls = joblib.load('finalized_model.sav')
    lis=[]
    lis.append(request.GET['baseline_value'])
    lis.append(request.GET['accelerations'])
    lis.append(request.GET['fetal_movement'])
    lis.append(request.GET['uterine_contractions'])
    lis.append(request.GET['light_decelerations'])
    lis.append(request.GET['severe_decelerations'])
    lis.append(request.GET['prolongued_decelerations'])
    lis.append(request.GET['abnormal_short_term_variability'])
    lis.append(request.GET['mean_value_of_short_term_variability'])
    lis.append(request.GET['percentage_of_time_with_abnormal_long_term_variability'])
    lis.append(request.GET['mean_value_of_long_term_variability'])
    lis.append(request.GET['histogram_width'])
    lis.append(request.GET['histogram_min'])
    lis.append(request.GET['histogram_max'])
    lis.append(request.GET['histogram_number_of_peaks'])
    lis.append(request.GET['histogram_number_of_zeroes'])
    lis.append(request.GET['histogram_mode'])
    lis.append(request.GET['histogram_mean'])
    lis.append(request.GET['histogram_median'])
    lis.append(request.GET['histogram_variance'])
    lis.append(request.GET['histogram_tendency'])

    print(lis)
    ans = cls.predict([lis])
    return render(request,"result.html",{'ans':ans ,'lis':lis})
