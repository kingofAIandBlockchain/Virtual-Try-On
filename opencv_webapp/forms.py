from django import forms
from .models import ImageUploadModel

class UploadImageForm(forms.Form):
    
    ps_image = forms.ImageField()
    cl_image = forms.ImageField()
    


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUploadModel
        fields = ('description', 'document' )