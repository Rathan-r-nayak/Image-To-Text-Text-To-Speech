from django.shortcuts import redirect, render
from PIL import Image
import pytesseract
from django.http import HttpResponse,HttpResponseRedirect
from django.template import loader
from .prediction import extractTextFromImage
import keras


msg = ""
image = None
# Create your views here.
def convertImageToText(request):
    context={
        "result":msg,
        "image":image
    }
    template = loader.get_template('convertToTextApp/index.html')
    return HttpResponse(template.render(context, request))


def extractText(request):
    if(request.method == 'POST'):
        try:
            uploaded_image = request.FILES['image']
            img = Image.open(uploaded_image)
            global image
            image = uploaded_image
            # print(keras.__version__)
            # text = pytesseract.image_to_string(img)
            # print("hi")
            text = extractTextFromImage(img)
            print("------------------------------------------------------------------------")
            print("-----------------------------Predicted Text-----------------------------")
            print(text)
            print("------------------------------------------------------------------------")
            global msg
            msg = text 
        except Exception as e:
           print(e)
        
    return redirect('convertImageToText')