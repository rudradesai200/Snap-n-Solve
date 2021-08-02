from django.contrib import sitemaps
from django.urls import reverse

class SudokuViewSitemap(sitemaps.Sitemap):
    priority = 0.9
    changefreq = 'weekly'

    def items(self):
        return ['Sudoku_home']

    def location(self, item):
        return reverse(item)
