from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from classification.apps import ClassificationConfig
import pandas as pd
import numpy as np
import text_process_prabhat99 as pp
import spacy

# Create your views here.
class News_Classification(APIView):
    def post(self, request, format=None):
        data = request.data
        h1 = data['user']['h1']
        h2 = data['user']['h2']

        data1 = []
        data1.append(h1)
        data1.append(h2)
        data = data1
        new_df = pd.DataFrame(data)
        new_df.columns = ['Text']
        new_df['Text'] = new_df['Text'].apply(lambda x : x.lower())
        new_df['Text'] = new_df['Text'].apply(lambda x : pp.cont_exp(x))
        new_df['Text'] = new_df['Text'].apply(lambda x : pp.remove_stopwords(x))
        new_df['Text'] = new_df['Text'].apply(lambda x : pp.remove_accented_chars(x))
        new_df['Text'] = new_df['Text'].apply(lambda x : pp.remove_special_chars(x))

        nlp = spacy.load('en_core_web_lg')

        new_df['vec'] = new_df['Text'].apply(lambda x : nlp(x).vector)

        X = new_df['vec'].to_numpy()
        X = X.reshape(-1,1)
        X = np.concatenate(np.concatenate(X, axis = 0), axis = 0).reshape(-1, 300)

        clf = ClassificationConfig.mlmodel
        output = {0:'business',4:'tech',2:'politics',3:'sport',1:'entertainment'}
        a = clf.predict(X)
        j = 0
        res = []
        for i in a:
            res.append(output[i])
        
        return Response(res, status = 200)
        # return Response([h1,h2], status = 200)

