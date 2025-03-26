from django.db import models

class WasteBin(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    description = models.TextField()

    def __str__(self):
        return f"Waste Bin at ({self.latitude}, {self.longitude})"

class WasteReport(models.Model):
    WASTE_CATEGORIES = [
        ('Plastic', 'Plastic'),
        ('Organic', 'Organic'),
        ('Metal', 'Metal'),
        ('Glass', 'Glass'),
        ('Paper', 'Paper'),
        ('Cardboard', 'Cardboard'),
        ('Trash', 'Trash'),
    ]

    latitude = models.FloatField()
    longitude = models.FloatField()
    waste_type = models.CharField(max_length=20, choices=WASTE_CATEGORIES)
    description = models.TextField()

    def __str__(self):
        return f"{self.waste_type} waste at ({self.latitude}, {self.longitude})"
