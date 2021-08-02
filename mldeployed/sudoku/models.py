from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

# Create your models here.
class Sudoku(models.Model):
    Sudoku_image = models.ImageField(upload_to='sudoku/sudoku/')
    status = models.BooleanField(default=True,null=True,blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

@receiver(post_delete, sender=Sudoku)
def submission_delete(sender, instance, **kwargs):
    instance.Sudoku_image.delete(False) 