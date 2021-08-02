from django import forms
from sudoku.models import Sudoku

class SudokuForm(forms.ModelForm):
    class Meta:
        model = Sudoku
        fields = ('Sudoku_image', )