#django imports
from django.shortcuts import render, redirect, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required

# Application imports
from sudoku.forms import SudokuForm
from sudoku.extract import parse_grid,show_digits
from sudoku.solution import start_solve

# Python Library
import os
import cv2
import numpy as np
from PIL import Image

# Create your views here.

def concate_images(request, grid, save_path):
    ims = 157
    maxs = ims*9
    new_im = Image.new('RGB',(maxs,maxs))
    for i in range(0,maxs,ims):
        for j in range(0,maxs,ims):
            im = Image.open(grid[j//ims][i//ims])
            new_im.paste(im,(i,j))
    new_im.save(settings.MEDIA_ROOT+save_path)

def generateimage(request,grid,save_path):
    folder_path = str(settings.BASE_DIR) + "/static/sudoku/digits/"
    img_grid = []
    for i in range(9):
        temp = []
        for j in range(9):
            temp.append(folder_path + "{}.png".format(grid[i][j]))
        img_grid.append(temp)
    concate_images(request,img_grid,save_path)

def home(request):
    uploaded_file_url = None
    extracted_file_url = None
    game= None
    if request.method == 'POST':
        form = SudokuForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()

            # Setting up paths and images
            uploaded_file_url = obj.Sudoku_image.url
            complete_image_url = str(settings.BASE_DIR)+uploaded_file_url
            file_name_with_extension = uploaded_file_url.split('/')[-1]
            extracted_file_url_complete = str(settings.BASE_DIR)+"/media/sudoku/extracted/"+file_name_with_extension
            file_name = file_name_with_extension.split(".")[0]
            extracted_file_url = "/media/sudoku/extracted/" + file_name_with_extension
            final_path = "/sudoku/solution/{}.png".format(file_name)

            f = True
            # Extract , solve and render
            try:
                input_grid,_ = parse_grid(request,complete_image_url,extracted_file_url_complete,False)
            except:
                messages.error(request,"Oops! Something went wrong. Our model cannot proeperly decode digits from the given sudoku image")
                form = SudokuForm()
                return render(request, 'sudoku/home.html', {
                    'form': form,
                })
            try:
                game = start_solve(input_grid)
            except:
                try:
                    input_grid,_ = parse_grid(request,complete_image_url,extracted_file_url_complete,True)
                except:
                    form = SudokuForm()
                    return render(request, 'sudoku/home.html', {
                        'form': form,
                    })
                try:
                    game = start_solve(input_grid)
                except:
                    game= None
            # print(input_grid)
            if game == None:
                obj.status = False
                if(f): messages.error(request,"Oops! Something went wrong. Our model cannot proeperly decode digits from the given sudoku image")
                obj.save()
            else:
                generateimage(request,game,final_path)
                return render(request, 'sudoku/index.html', {
                    'form': form,
                    'uploaded_file_url':uploaded_file_url,
                    'extracted_file_url':extracted_file_url,
                    "solution_file_url":"/media" + final_path
                })
        else:
            messages.error(request,"Only Image file formats are allowed.")
    else:
        form = SudokuForm()

    return render(request, 'sudoku/home.html', {
        'form': form,
    })

@staff_member_required
def remove_files(request):
    media_dir = settings.MEDIA_ROOT
    extracted_removed = []
    for fil in os.listdir(media_dir + "/sudoku/extracted"):
        os.remove(os.path.join(media_dir+"/sudoku/extracted/"+fil))
        extracted_removed.append("/sudoku/extracted/"+fil)
    solutions_removed = []
    for fil in os.listdir(media_dir + "/sudoku/solution"):
        os.remove(os.path.join(media_dir+"/sudoku/solution/"+fil))
        solutions_removed.append("/sudoku/solution/"+fil)
    objret = "<h2>Extracted</h2><br><ol>"
    for x in extracted_removed:
        objret += "<li>" + x
    objret += "</ol>"
    objret += "<h2>Soltution</h2><br><ol>"
    for x in solutions_removed:
        objret += "<li>" + x
    objret += "</ol>"
    return HttpResponse(objret)