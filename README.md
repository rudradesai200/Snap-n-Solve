<a href="https://www.buymeacoffee.com/rudradesai200" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

# MLDeployed
MLDeployed is a project designed to solve sudoku problems directly from their images.

## Setup
  - This project requires Python, Django, Tensorflow, OpenCV and Pillow.
  - All the pip requirements are mentioned in requirements.txt
  - Use `pip install -r requirements.txt` to install them

## Usage
  - To start the server, go to the mldeployed folder.
  - Run `python3 manage.py makemigrations sudoku` to create the database table migrations
  - Then, Run `python3 manage.py migrate` to migrate the tables and create a sqlite instance
  - Finally, run `python3 manage.py runserver` to get a Django server running on your local machine