import os
import uuid

import streamlit as st
from PIL import Image

from config import INPUT_IMAGE_DOWNLOAD_PATH
from libs.utils.ml_model.src.helpers import image_predict_helpers

st.title('Find your celebrity twin.')


class PlatformWebappService:
    @staticmethod
    def generate_phases_for_response(
        input_image_path: str,
        celebrity_name: str
    ):
        celeb_img_path = image_predict_helpers.get_celebrity_image_path(
            celebrity_name
        )
        col1, col2 = st.columns(2)
        with col1:
            image_predict_helpers.extract_face_from_image(
                input_image_path, input_image_path, detector_backend='mtcnn'
            )
            st.image(input_image_path, width=150, caption='Input Image')
        with col2:
            overlay_image_path = (
                image_predict_helpers.overlay_images_for_phases(
                    input_image_path, celeb_img_path, tranperncy=0.3
                ))
            st.image(overlay_image_path, width=150)
            os.remove(overlay_image_path)
        col3, col4 = st.columns(2)
        with col3:
            overlay_image_path = (
                image_predict_helpers.overlay_images_for_phases(
                    celeb_img_path, input_image_path, tranperncy=0.3
                ))
            st.image(overlay_image_path, width=150)
            os.remove(input_image_path)
            os.remove(overlay_image_path)
        with col4:
            st.image(
                celeb_img_path, width=150,
                caption=f'You look like {celebrity_name}'
            )

    def get_similarity_scores_from_image_url(self):
        image_url = st.text_input('Enter link to your jpg image.')
        if image_url != '':
            try:
                input_image_path = (
                    image_predict_helpers.download_image_and_get_image_path(
                        image_url
                    ))
                similiarities = image_predict_helpers.get_similarity_scores(
                    input_image_path
                )
                st.text(similiarities)
                celebrity_name = list(similiarities.keys())[0]
                self.generate_phases_for_response(
                    input_image_path, celebrity_name
                )
            except Exception as error:
                st.text(f'Invalid image url, Error: {error}')

    def generate_user_response_from_image_path(self, input_image):
        input_image = Image.open(input_image)
        input_image_path = f'{INPUT_IMAGE_DOWNLOAD_PATH}/{uuid.uuid4()}.jpg'
        input_image.save(input_image_path)
        similiarities = image_predict_helpers.get_similarity_scores(
            input_image_path
        )
        celebrity_name = list(similiarities.keys())[0]
        self.generate_phases_for_response(
            input_image_path, celebrity_name
        )

    def get_similarity_scores_from_image(self):
        input_image = st.file_uploader("Choose a file")
        if input_image is not None:
            try:
                self.generate_user_response_from_image_path(input_image)
            except Exception as error:
                print(error)
                st.text(
                    'Invalid image!! Make sure your face is visible while '
                    'taking a picture.'
                )

    def get_similarity_scores_by_taking_picture(self):
        input_image = st.camera_input(
            label='Take your picture to get your celebrity look a like.'
        )
        if input_image is not None:
            try:
                self.generate_user_response_from_image_path(input_image)
            except Exception as error:
                print(error)
                st.text(
                    'Invalid image!! Make sure your face is visible while '
                    'taking a picture.'
                )

    def launch_web_app(self):
        st.sidebar.info('Please choose a option.')
        user_choice = st.sidebar.selectbox(
            'Options',
            ['Picture url', 'Upload picture', 'Take Picture']
        )
        if user_choice == 'Picture url':
            self.get_similarity_scores_from_image_url()
        elif user_choice == 'Upload picture':
            self.get_similarity_scores_from_image()
        elif user_choice == 'Take Picture':
            self.get_similarity_scores_by_taking_picture()


platform_webapp_service = PlatformWebappService()
if __name__ == '__main__':
    platform_webapp_service.launch_web_app()
