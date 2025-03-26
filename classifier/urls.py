from django.urls import path
from classifier import views
urlpatterns = [
    path('', views.home, name='home'),
    path('predict_batch/', views.predict_batch, name='predict_batch'),
    path('map/', views.waste_bin_map, name='waste_bin_map'),
    path('get-waste-bins/', views.get_waste_bins, name='get_waste_bins'),
    path('update-bin-location/', views.update_bin_location, name='update_bin_location'),
    path('add-waste-bin/', views.add_waste_bin, name='add_waste_bin'),
    path('add-waste-report/', views.add_waste_report, name='add_waste_report'),
    path('get-waste-reports/', views.get_waste_reports, name='get_waste_reports'),
]
