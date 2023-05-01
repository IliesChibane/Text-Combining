from django.db import models

class Results(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    task = models.CharField(max_length=100, blank=True, default='test')
    result = models.CharField(max_length=100, blank=True, default='')

class Inputs(models.Model):
    provider = models.CharField(max_length=100, blank=True, default='')
    input_text = models.CharField(max_length=100, blank=True, default='')
    output_text = models.ForeignKey("AdvancedGenius1.Results", on_delete=models.CASCADE)
