"""carbon_be_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from carbon_be_project import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns 


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('input_predict/', views.input_predict, name='input_predict'),
    path('result/', views.result, name='result'),
    path('compare/', views.compare, name='compare'),
    path('visualize/',views.visualize,name='visualize'),
    path('solution/',views.solution,name='solution'),
    path('input_compare/',views.input_compare,name='input_compare'),
]
urlpatterns +=staticfiles_urlpatterns()