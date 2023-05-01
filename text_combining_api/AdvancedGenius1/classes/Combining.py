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

            result = {
                'task': task,
                'result': combined_sentence
            }

            # Create a result serializer
            result_serializer = ResultsSerializer(data=result)

            if result_serializer.is_valid(): # if the result is valid
                result_serializer.save() # save the result in the database
                # We save the inputs as well
                id_result = result_serializer.data["id"] # get the id of the result
                providers = request.data['providers'] # get the providers

                i = 0
                b = True
                all_inputs = []
                for text in texts:
                    # Create an input object
                    inputs = {
                        'provider': providers[i],
                        'input_text': text,
                        'output_text': id_result
                    }
                    i += 1
                    input_serializer = InputsSerializer(data=inputs) # Create an input serializer
                    if input_serializer.is_valid(): # if the input is valid
                        input_serializer.save() # save the input in the database
                        all_inputs.append(input_serializer.data) # add the input to the list of inputs
                    else:
                        b = False # if the input is not valid, we set b to False
                if b: # if all the inputs are valid
                    # We return the result and the inputs
                    final_result = {
                        "inputs" : all_inputs,
                        "result" : result_serializer.data
                    }
                    return JsonResponse(final_result, status=status.HTTP_201_CREATED)
                else: # if one of the inputs is not valid
                    # We send an error message
                    return JsonResponse(result_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            else: # if the result is not valid
                # We send an error message
                return JsonResponse(result_serializer.errors, status=status.HTTP_400_BAD_REQUEST)