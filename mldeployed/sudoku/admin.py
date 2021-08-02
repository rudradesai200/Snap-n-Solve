from django.contrib import admin
from sudoku.models import *

class SudokuAdmin(admin.ModelAdmin):
    list_display = ('id','status',)

admin.site.register(Sudoku,SudokuAdmin)
# Register your models here.
