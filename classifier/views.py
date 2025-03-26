

import os
import numpy as np
import tensorflow as tf
import cv2  # Using OpenCV for consistent image handling
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

# ✅ Load the trained AI model (Disable compilation for safety)
model_path = os.path.join(settings.BASE_DIR, 'waste_classifier_mobilenet.h5')
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ Waste categories (Ensure they match the training labels)
CATEGORIES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def home(request):
    return render(request, 'classifier/home.html')

def preprocess_image(image_path):
    """ ✅ Preprocess image exactly like in Colab """
    img = cv2.imread(image_path)  # Read using OpenCV
    if img is None:
        print(f"⚠ Error: Could not read image {image_path}")
        return None  # Skip processing if image couldn't be loaded
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize (same as Colab)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_batch(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        result = []
        disposal_tips = {
            'Cardboard': "Flatten the cardboard and recycle it in a dry state.",
            'Glass': "Dispose of broken glass separately and recycle glass bottles.",
            'Metal': "Recycle aluminum and steel cans; avoid mixing with plastic.",
            'Paper': "Keep paper dry and recycle it properly.",
            'Plastic': "Check the recycling symbol; avoid single-use plastics.",
            'Trash': "Dispose of non-recyclables in the trash bin properly."
        }

        for img_file in request.FILES.getlist('images'):
            img_path = os.path.join(settings.MEDIA_ROOT, img_file.name)

            # ✅ Save uploaded image
            with open(img_path, 'wb+') as destination:
                for chunk in img_file.chunks():
                    destination.write(chunk)

            # ✅ Preprocess the image
            img_array = preprocess_image(img_path)
            if img_array is None:
                continue  # Skip if preprocessing fails

            # ✅ Predict the waste category
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = CATEGORIES[predicted_index]  
            confidence = round(float(np.max(prediction)) * 100, 2)  # Convert to percentage

            # ✅ Store results for each image
            result.append({
                'image_url': settings.MEDIA_URL + img_file.name,
                'prediction': predicted_class,
                'confidence': confidence,
                'disposal_tip': disposal_tips.get(predicted_class, "Dispose of properly.")
            })

        return JsonResponse(result, safe=False)

    return JsonResponse({'error': 'No images uploaded'}, status=400)



from django.shortcuts import render

def waste_bin_map(request):
    return render(request, 'classifier/map.html')


from django.http import JsonResponse
from .models import WasteBin  # Ensure this model exists

def get_waste_bins(request):
    bins = WasteBin.objects.all()
    data = [{'id': bin.id, 'lat': bin.latitude, 'lon': bin.longitude, 'description': bin.description} for bin in bins]
    return JsonResponse(data, safe=False)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import WasteBin  # Ensure your model is correct

@csrf_exempt  # Allow AJAX POST requests
def update_bin_location(request):
    if request.method == 'POST':
        bin_id = request.POST.get('bin_id')
        lat = request.POST.get('lat')
        lon = request.POST.get('lon')

        try:
            bin_obj = WasteBin.objects.get(id=bin_id)
            bin_obj.latitude = lat
            bin_obj.longitude = lon
            bin_obj.save()
            return JsonResponse({'status': 'success', 'message': 'Location updated!'})
        except WasteBin.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Bin not found!'}, status=404)

    return JsonResponse({'status': 'error', 'message': 'Invalid request!'}, status=400)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import WasteBin

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import WasteBin, WasteReport

@csrf_exempt
def add_waste_report(request):
    """Allow users to add waste reports manually"""
    if request.method == 'POST':
        lat = request.POST.get('lat')
        lon = request.POST.get('lon')
        waste_type = request.POST.get('waste_type')
        description = request.POST.get('description', '')

        new_report = WasteReport.objects.create(latitude=lat, longitude=lon, waste_type=waste_type, description=description)
        return JsonResponse({'status': 'success', 'message': 'Waste report added!', 'report_id': new_report.id})

    return JsonResponse({'status': 'error', 'message': 'Invalid request!'}, status=400)

def get_waste_reports(request):
    """Fetch waste reports with optional filtering"""
    waste_type = request.GET.get('waste_type')  # Optional filter
    reports = WasteReport.objects.all()

    if waste_type:
        reports = reports.filter(waste_type=waste_type)

    data = [{'id': report.id, 'lat': report.latitude, 'lon': report.longitude, 'waste_type': report.waste_type, 'description': report.description} for report in reports]
    return JsonResponse(data, safe=False)

@csrf_exempt
def add_waste_bin(request):
    if request.method == 'POST':
        lat = request.POST.get('lat')
        lon = request.POST.get('lon')
        description = request.POST.get('description')

        # ✅ Create and save new waste bin
        new_bin = WasteBin.objects.create(latitude=lat, longitude=lon, description=description)
        return JsonResponse({'status': 'success', 'message': 'Waste bin added!', 'bin_id': new_bin.id})

    return JsonResponse({'status': 'error', 'message': 'Invalid request!'}, status=400)
