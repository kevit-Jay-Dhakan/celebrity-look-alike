from pymongo import MongoClient

from libs.utils.db.mongoose.src.config import DATABASE_NAME, DATABASE_URI

client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
