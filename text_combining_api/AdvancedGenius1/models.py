from django.db import models

class Results(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    task = models.CharField(max_length=100, blank=True, default='test')
    inputs = models.JSONField(null=True)
    result = models.TextField(blank=True, default='')
