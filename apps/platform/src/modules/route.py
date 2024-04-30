import time

from fastapi import APIRouter, Depends, File, UploadFile

from apps.platform.src.modules.dto import (
    DeleteCelebrityEmbeddingsCelebNamesModel,
    GenerateCelebrityEmbeddingsCelebNamesModel,
    GenerateCelebrityEmbeddingsModel,
    GetCelebrityLookalikeModel
)
from apps.platform.src.modules.service import platform_service
from libs.utils.ml_model.src.config import INPUT_IMAGE_DOWNLOAD_PATH

platform_route = APIRouter(tags=['Platform'], prefix='/predict')


class PlatformDownloads:
    @staticmethod
    @platform_route.post('/')
    def get_celebrity_lookalike(
        user_data: GetCelebrityLookalikeModel
    ):
        start_time = time.perf_counter()
        return {
            'response': platform_service.fetch_celebrity_lookalike(
                user_data=user_data
            ),
            'Time Taken': time.perf_counter() - start_time
        }

    @staticmethod
    @platform_route.post('/upload_image')
    async def get_celebrity_lookalike_by_image_upload(
        image: UploadFile = File(...)
    ):
        start_time = time.perf_counter()
        contents = await image.read()
        image_path = f'{INPUT_IMAGE_DOWNLOAD_PATH}/{image.filename}'
        with open(image_path, "wb") as image_file:
            image_file.write(contents)
        return {
            'response': platform_service.fetch_celebrity_lookalike(
                image_path=image_path
            ),
            'Time Taken': time.perf_counter() - start_time
        }

    @staticmethod
    @platform_route.post('/generate_embeddings/file_upload')
    def generate_and_store_embeddings(
        input_data: GenerateCelebrityEmbeddingsModel = Depends(),
        zip_file_of_images: UploadFile = File(...)
    ):
        start_time = time.perf_counter()
        return {
            'response': platform_service.generate_embeddings_from_zip_file(
                input_data, zip_file_of_images
            ),
            'Time Taken': time.perf_counter() - start_time
        }

    @staticmethod
    @platform_route.post('/generate_embeddings/celeb_names')
    def generate_and_store_embeddings_from_celeb_names(
        input_data: GenerateCelebrityEmbeddingsCelebNamesModel
    ):
        start_time = time.perf_counter()
        return {
            'response': platform_service.generate_embeddings_from_celeb_names(
                input_data
            ),
            'Time Taken': time.perf_counter() - start_time
        }

    @staticmethod
    @platform_route.get('/')
    def get_list_of_celeb_names(
    ):
        return {'response': platform_service.get_celeb_names_from_database()}

    @staticmethod
    @platform_route.delete('/delete')
    def delete_celeb_data_from_database(
        input_data: DeleteCelebrityEmbeddingsCelebNamesModel
    ):
        start_time = time.perf_counter()
        return {
            'response': platform_service.delete_image_and_embeddings_of_celeb(
                input_data
            ),
            'Time Taken': time.perf_counter() - start_time
        }
