from django.urls import path
from .views import EdenCombiningView

urlpatterns = [
    path('AdvancedGenius1', EdenCombiningView.text_combining),
    ]