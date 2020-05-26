from django.db import models

# Create your models here.
class MyData(models.Model):
    Name= models.CharField(max_length=30,default='Abc')
    Last_Name= models.CharField(max_length=30,default='Xyz')
    Mobile=models.IntegerField()
    Email= models.CharField(max_length=30,default='abc@gmail.com')
    Gender=models.CharField(max_length=15,default='Male')
    TOFEL=models.IntegerField()
    GRE=models.IntegerField()
    UNI_rating=models.IntegerField()
    SOP=models.IntegerField()
    LOR=models.IntegerField()
    CGPA=models.IntegerField()
    Research_Ex=models.IntegerField()
    
