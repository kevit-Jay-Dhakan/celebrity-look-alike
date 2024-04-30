import zipfile
from os import listdir, path, remove
from os.path import join
from shutil import copyfileobj, rmtree

from apps.platform.src.modules.dto import (
    DeleteCelebrityEmbeddingsCelebNamesModel,
    GenerateCelebrityEmbeddingsCelebNamesModel,
    GenerateCelebrityEmbeddingsModel,
    GetCelebrityLookalikeModel
)
from libs.utils.ml_model.src.config import (
    TRAIN_IMAGES_FOLDER_PATH
)
from libs.utils.ml_model.src.helpers import image_predict_helpers
from libs.utils.ml_model.src.repository import celebrities_embeddings_repository


class PlatformService:
    @staticmethod
    def unzip_file(zip_file, extract_to):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    @staticmethod
    def fetch_celebrity_lookalike(
        user_data: GetCelebrityLookalikeModel = None,
        image_path: str = None
    ):
        try:
            if image_path is None:
                image_path = (
                    image_predict_helpers.download_image_and_get_image_path(
                        user_data.imageUrl
                    ))
            similarities = image_predict_helpers.get_similarity_scores(
                image_path
            )
            return {'similaritiesScore': similarities}
        except Exception as error:
            return {'Error': str(error)}

    def generate_embeddings_from_zip_file(
        self, input_data: GenerateCelebrityEmbeddingsModel, zip_file_of_images
    ):
        detector_backend = input_data.detectorBackend
        model_name = input_data.modelName
        try:
            with open(zip_file_of_images.filename, "wb") as buffer:
                copyfileobj(zip_file_of_images.file, buffer)

            source_dir = str(zip_file_of_images.filename).replace('.zip', '')
            self.unzip_file(zip_file_of_images.filename, source_dir)
            file_list = listdir(source_dir)

            source_images_path = list()
            for file_name in file_list:
                if file_name.endswith(
                    ('.jpg', '.jpeg', '.png', '.gif', '.webp')
                ):
                    source_path = join(source_dir, file_name)
                    source_images_path.append(source_path)
            result = image_predict_helpers.generate_and_store_embeddings(
                source_images_path, detector_backend, model_name,
            )
            remove(zip_file_of_images.filename)
            rmtree(source_dir)
            return result
        except Exception as error:
            print(error)
            return 'Failed to generate embeddings for celebrity images.'

    @staticmethod
    def generate_embeddings_from_celeb_names(
        input_data: GenerateCelebrityEmbeddingsCelebNamesModel,
    ):
        return (
            image_predict_helpers
            .generate_and_store_embeddings_from_celeb_names(
                input_data.celebNames
            ))

    @staticmethod
    def get_celeb_names_from_database():
        celeb_names = list(
            celebrities_embeddings_repository.find(
                {}, {'_id': 0, 'name': 1}
            )
        )
        return {
            'totalCelebCount': len(celeb_names),
            'celebNames': [doc['name'] for doc in celeb_names]
        }

    @staticmethod
    def delete_image_and_embeddings_of_celeb(
        input_data: DeleteCelebrityEmbeddingsCelebNamesModel
    ):
        celeb_names = input_data.celebNames
        [
            remove(join(TRAIN_IMAGES_FOLDER_PATH, f'{celeb_name}.jpg'))
            for celeb_name in celeb_names
            if path.exists(join(TRAIN_IMAGES_FOLDER_PATH, f'{celeb_name}.jpg'))
        ]
        result = celebrities_embeddings_repository.delete_many(
            {'name': {'$in': celeb_names}}
        )
        return (
            f"Successfully deleted embeddings of {result.deleted_count} celebs "
            f"from database."
        )


platform_service = PlatformService()
