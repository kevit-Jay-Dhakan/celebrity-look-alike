from libs.utils.db.mongoose.src import BaseRepository
from libs.utils.ml_model.src.collection import celebrities_embeddings_collection


class DeliveryRepository(BaseRepository):
    def __init__(self):
        super().__init__(collection=celebrities_embeddings_collection)


celebrities_embeddings_repository = DeliveryRepository()
