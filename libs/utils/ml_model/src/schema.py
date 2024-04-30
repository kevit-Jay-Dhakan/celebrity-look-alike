CELEBERITY_EMBEDDINGS_VALIDATION_SCHEMA = {
    'bsonType': 'object',
    'properties': {
        '_id': {'bsonType': 'objectId'},
        'name': {'bsonType': 'string'},
        'modelUsed': {'bsonType': 'string'},
        'embedding': {'bsonType': 'array'},
        'facial_area': {'bsonType': 'object'},
        'face_confidence': {'bsonType': 'double'},
        'createdAt': {'bsonType': 'date'},
        'updatedAt': {'bsonType': 'date'}
    }
}
