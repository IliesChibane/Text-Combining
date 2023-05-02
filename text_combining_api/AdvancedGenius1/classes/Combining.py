from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.permissions import AllowAny


from .Text_mining.Algorithm import text_mining_algorithm
from .SSA.similarity import similarity_analysis

from ..serializers import ResultsSerializer

import numpy as np

class EdenCombiningView(APIView):
    permission_classes=[AllowAny]

    @api_view(['POST'])
    def text_combining(request):
        if request.method == 'POST':

            # fetch the text inputs
            texts = request.data['sentences']
            sentences = []
            for t in texts:
                sentences.append([t])
            sentences = np.array(sentences)
            
            # Apply the text combining algorithm
            masked_sentence, uncommon_words = text_mining_algorithm(sentences)

            combined_sentence = similarity_analysis(masked_sentence, uncommon_words)

            # Save the result in the database
            # Create a result object
            task = request.data['task']

            providers = request.data['providers'] # get the providers

            inputs = dict()

            for i in range(len(texts)):
                inputs["Input "+str(i+1)] = {
                    'provider': providers[i],
                    'input_text': texts[i]
                } 

            result = {
                'task': task,
                'inputs': inputs,
                'result': combined_sentence
            }

            # Create a result serializer
            result_serializer = ResultsSerializer(data=result)

            if result_serializer.is_valid(): # if the result is valid
                result_serializer.save() # save the result in the database
                return JsonResponse(result_serializer.data, status=status.HTTP_201_CREATED)
            else: # if the result is not valid
                # We send an error message
                return JsonResponse(result_serializer.errors, status=status.HTTP_400_BAD_REQUEST)