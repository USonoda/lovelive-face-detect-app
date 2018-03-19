from flask import Blueprint
app = Blueprint("tmp", __name__,
    static_url_path='/tmp', static_folder='./tmp'
)