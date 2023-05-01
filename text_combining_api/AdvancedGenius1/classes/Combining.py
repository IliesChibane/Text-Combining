from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.permissions import AllowAny


from .Text_mining.Algorithm import text_mining_algorithm
from .SSA.similarity import similarity_analysis

from ..serializers import ResultsSerializer, InputsSerializer

import numpy as np

class EdenCombiningView(APIView):
    permission_classes=[AllowAny]

    @api_view(['POST'])
    def text_combining(request):
        if request.method == 'POST':
            texts = request.data['sentences']
            sentences = []
            for t in texts:
                sentences.append([t])
            sentences = np.array(sentences)
            
            masked_sentence, uncommon_words = text_mining_algorithm(sentences)

            combined_sentence = similarity_analysis(masked_sentence, uncommon_words)

            task = request.data['task']

            result = {
                'task': task,
                'result': combined_sentence
            }

            result_serializer = ResultsSerializer(data=result)

            if result_serializer.is_valid():
                result_serializer.save()
                id_result = result_serializer.data["id"]
                providers = request.data['providers']
                i = 0
                b = True
                all_inputs = []
                for text in texts:
                    inputs = {
                        'provider': providers[i],
                        'input_text': text,
                        'output_text': id_result
                    }
                    i += 1
                    input_serializer = InputsSerializer(data=inputs)
                    if input_serializer.is_valid():
                        input_serializer.save()
                        all_inputs.append(input_serializer.data)
                    else:
                        b = False
                if b:
                    final_result = {
                        "inputs" : all_inputs,
                        "result" : result_serializer.data
                    }
                    return JsonResponse(final_result, status=status.HTTP_201_CREATED)
                else:
                    return JsonResponse(result_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else:
                return JsonResponse(result_serializer.errors, status=status.HTTP_400_BAD_REQUEST)