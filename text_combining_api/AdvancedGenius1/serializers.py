from rest_framework import serializers 
from .models import *

class ResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Results
        fields = ('id', 'task', 'result')

class InputsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Inputs
        fields = ('id', 'provider', 'input_text', 'output_text')
