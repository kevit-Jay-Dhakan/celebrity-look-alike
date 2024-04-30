import itertools
import uuid
from operator import itemgetter
from os import makedirs
from os.path import join
from shutil import rmtree
from typing import List

import cv2
import requests
from deepface import DeepFace
from matplotlib import pyplot as plt
from pandas import read_parquet

from libs.utils.common.src.modules.string_helpers import apply_unidecode
from libs.utils.ml_model.src.config import (
    INPUT_IMAGE_DOWNLOAD_PATH,
    TRAIN_IMAGES_FOLDER_PATH
)
from libs.utils.ml_model.src.repository import celebrities_embeddings_repository


class ImagePredictHelpers:
    @staticmethod
    def download_image_and_get_image_path(image_url: str):
        image_name = image_url.split('/')[-1]
        image_path = f'{INPUT_IMAGE_DOWNLOAD_PATH}/{image_name}'
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded successfully and saved as {image_path}")
        except Exception as error:
            print(error)
            raise Exception(
                "Failed to download image from image url. "
                "Please provide a valid and openly available image url."
            )
        return image_path

    @staticmethod
    def extract_face_from_image(
        input_image_path: str,
        output_image_path: str,
        target_size: tuple[int, int] = (200, 200),
        detector_backend: str = 'dlib'
    ):
        detected_face = DeepFace.extract_faces(
            img_path=input_image_path,
            target_size=target_size,
            detector_backend=detector_backend
        )
        plt.imshow(detected_face[0]['face'])
        plt.axis('off')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

    def generate_and_store_embeddings(
        self,
        train_images_path: List[str],
        detector_backend: str = 'dlib',
        model_name: str = 'Facenet512'
    ):
        count = 0
        repeat_celeb_count = 0
        for image_path in train_images_path:
            try:
                celebrity_name = image_path.split('/')[-1].split('.')[0]
                if celebrities_embeddings_repository.find_one(
                    {'name': celebrity_name}
                ):
                    print('Celebrity image already present in database.')
                    repeat_celeb_count += 1
                    continue
                celebrity_features = DeepFace.represent(
                    image_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False
                )
                dominant_gender = self.get_gender_from_image_path(image_path)
                if (
                    (len(celebrity_features) != 0) and
                    ('embedding' in celebrity_features[0])
                ):
                    celebrities_embeddings_repository.insert_one(
                        {
                            'name': celebrity_name,
                            'dominantGender': dominant_gender,
                            'embedding': celebrity_features[0]['embedding'],
                            'facialArea': celebrity_features[0]['facial_area'],
                            'faceConfidence': celebrity_features[0][
                                'face_confidence'],
                            'modelUsed': model_name
                        }
                    )
                    destination_path = join(
                        TRAIN_IMAGES_FOLDER_PATH, f'{celebrity_name}.jpg'
                    )
                    self.extract_face_from_image(
                        image_path, destination_path,
                        detector_backend=detector_backend
                    )
                    count += 1
            except Exception as error:
                print(error)
                raise Exception(f'Failed to generate embeddings for images.')
        return (
            f"Successfully generated and stored embeddings of {count} "
            f"images in database. And embeddings of {repeat_celeb_count} "
            f"celebs were already there in database."
        )

    def get_similarity_scores(self, input_image_path: str):
        try:
            user_features = DeepFace.represent(
                input_image_path, model_name="Facenet512",
                detector_backend="dlib"
            )
            user_embeddings = user_features[0]['embedding']
            user_gender = self.get_gender_from_image_path(input_image_path)
            similarities = dict()
            for celebrity in celebrities_embeddings_repository.find(
                {'dominantGender': user_gender}
            ):
                similarity = DeepFace.verify(
                    user_embeddings, celebrity['embedding'],
                    model_name="Facenet512", detector_backend="dlib",
                    distance_metric="cosine"
                )
                if similarity['distance']:
                    similarities[celebrity['name']] = similarity['distance']
            similarities = dict(
                sorted(similarities.items(), key=itemgetter(1))
            )
            return dict(itertools.islice(similarities.items(), 5))
        except Exception as error:
            print(error)
            raise Exception("Failed to get your celebrity lookalike.")

    @staticmethod
    def get_celebrity_image_path(celebrity_name: str):
        image_path = join(TRAIN_IMAGES_FOLDER_PATH, f'{celebrity_name}.jpg')
        return image_path

    @staticmethod
    def overlay_images_for_phases(img1, img2, tranperncy: float = 0.3):
        background = cv2.imread(img1)
        overlay_image = cv2.imread(img2)

        overlay_image = cv2.resize(overlay_image, background.shape[1::-1])

        result = cv2.addWeighted(
            background, 1 - tranperncy, overlay_image, tranperncy, 0
        )
        overlay_image_path = join(
            INPUT_IMAGE_DOWNLOAD_PATH, f'{uuid.uuid4()}.jpg'
        )
        cv2.imwrite(overlay_image_path, result)
        return overlay_image_path

    def generate_and_store_embeddings_from_celeb_names(
        self, celeb_list: List[str]
    ):
        dataset_path = 'libs/utils/ml_model/celebs_dataset.parquet'
        output_dir = f'libs/utils/ml_model/{uuid.uuid4()}'
        celebs_df = read_parquet(dataset_path, columns=['name', 'image'])

        celebs_df['name'] = celebs_df['name'].apply(apply_unidecode)
        filtered_celeb_list = celebs_df[celebs_df['name'].isin(celeb_list)]
        print(
            f'Total {len(filtered_celeb_list)} celebs found in dataset'
            f' from your input celebs list.'
        )
        if len(filtered_celeb_list) > 0:
            images_path = []
            for index, row in filtered_celeb_list.iterrows():
                makedirs(output_dir, exist_ok=True)
                image_name = f"{apply_unidecode(row['name'])}.jpg"
                image_path = join(output_dir, image_name)
                with open(image_path, 'wb') as f:
                    f.write(row['image']['bytes'])
                images_path.append(image_path)
            print("Images downloaded successfully.")
            result = self.generate_and_store_embeddings(
                train_images_path=images_path
            )
            rmtree(output_dir, ignore_errors=True)
            return result
        return 'None of the celebrity from your list is present in the dataset.'

    @staticmethod
    def get_gender_from_image_path(
        input_image_path: str = None,
        detector_backend: str = 'mtcnn',
        suppress_logs: bool = True
    ):
        try:
            features = DeepFace.analyze(
                input_image_path,
                actions=['gender'],
                detector_backend=detector_backend,
                silent=suppress_logs
            )
            dominant_gender = features[0]['dominant_gender']
            return dominant_gender
        except Exception as error:
            print(f'Failed to get dominant gender {str(error)}')
            return None


image_predict_helpers = ImagePredictHelpers()
if __name__ == '__main__':
    docs = list(celebrities_embeddings_repository.find({}, {}))
