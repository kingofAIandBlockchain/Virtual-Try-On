from django.shortcuts import render
from django.shortcuts import redirect
from .forms import UploadImageForm
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from django.conf import settings
from .opencv_dface import opencv_dface


# Create your views here.
def first_view(request):
    return render(request, 'opencv_webapp/first_view.html', {})


def uimage(request):
    if request.method == 'POST':
        form = UploadImageForm(request.FILES, request.FILES)
        
        
        if form.is_valid():
            cl_file = request.FILES['cl_image']
            ps_file = request.FILES['ps_image']
            fs = FileSystemStorage()
            cl_filename = fs.save(cl_file.name, cl_file)
            ps_filename = fs.save(ps_file.name, ps_file)
            
            cl_path = settings.MEDIA_URL + cl_filename
            ps_path = settings.MEDIA_URL + ps_filename

            cl = settings.MEDIA_ROOT_URL + cl_path
            ps = settings.MEDIA_ROOT_URL + ps_path
            
            res = opencv_dface(cl, ps)
            
            return render(request, 'opencv_webapp/uimage.html', {'form': form, 'uploaded_file_url': res})
                
    else:
        form = UploadImageForm()
        return render(request, 'opencv_webapp/uimage.html', {'form': form})


