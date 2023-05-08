from rest_framework import serializers 
from .models import *

class ResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Results
        fields = ('id', 'task', 'inputs', 'result')
