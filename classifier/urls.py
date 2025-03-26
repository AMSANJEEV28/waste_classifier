from django.urls import path
from .views import (
    home, predict_batch, waste_bin_map, get_waste_bins, update_bin_location, add_waste_bin, 
    add_waste_report, get_waste_reports
)

urlpatterns = [
    path('', home, name='home'),
    path('predict_batch/', predict_batch, name='predict_batch'),
    path('map/', waste_bin_map, name='waste_bin_map'),
    path('get-waste-bins/', get_waste_bins, name='get_waste_bins'),
    path('update-bin-location/', update_bin_location, name='update_bin_location'),
    path('add-waste-bin/', add_waste_bin, name='add_waste_bin'),
    path('add-waste-report/', add_waste_report, name='add_waste_report'),  # ✅ New endpoint
    path('get-waste-reports/', get_waste_reports, name='get_waste_reports'),  # ✅ New endpoint
]
