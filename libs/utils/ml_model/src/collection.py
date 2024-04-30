import pymongo

from libs.utils.db.mongoose.src import db
from libs.utils.ml_model.src.schema import \
    CELEBERITY_EMBEDDINGS_VALIDATION_SCHEMA

validation_options = {
    'validator': {'$jsonSchema': CELEBERITY_EMBEDDINGS_VALIDATION_SCHEMA},
    'validationLevel': 'strict',
    'validationAction': 'warn'
}

CELEBERITY_EMBEDDINGS_COLLECTION_NAME = "celebrities_embeddings"
celebrities_embeddings_collection = db[CELEBERITY_EMBEDDINGS_COLLECTION_NAME]

celebrities_embeddings_collection.create_index(
    [("embedding", pymongo.ASCENDING)]
)

celebrities_embeddings_collection.database.command(
    'collMod',
    celebrities_embeddings_collection.name,
    **validation_options
)
