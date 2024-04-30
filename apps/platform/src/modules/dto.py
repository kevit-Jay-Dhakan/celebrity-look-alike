from typing import List

from pydantic import BaseModel, Field, model_validator


class GetCelebrityLookalikeModel(BaseModel):
    imageUrl: str = Field(
        description='Enter link to your jpg image.',
        example='https://upload.wikimedia.org/wikipedia/commons/c/c1'
                '/Lionel_Messi_20180626.jpg',
        min_length=1
    )

    @model_validator(mode='after')
    def post_validate(self):
        self.imageUrl = self.imageUrl.strip()
        if self.imageUrl == '':
            raise ValueError('Image url cannot be empty string.')
        return self


class GenerateCelebrityEmbeddingsModel(BaseModel):
    detectorBackend: str = Field(
        description='Face detector backend',
        example='mtcnn',
        min_length=1,
        default='mtcnn'
    )
    modelName: str = Field(
        description='Name of model to use for generating embeddings.',
        example='opencv',
        min_length=1,
        default='Facenet512'
    )

    @model_validator(mode='after')
    def post_validate(self):
        self.detectorBackend = str(self.detectorBackend).strip()
        self.modelName = str(self.modelName).strip()
        if self.detectorBackend == '':
            raise ValueError('detectorBackend cannot be empty string.')
        if self.modelName == '':
            raise ValueError('modelName cannot be empty string.')
        return self


class GenerateCelebrityEmbeddingsCelebNamesModel(
    GenerateCelebrityEmbeddingsModel
):
    celebNames: List[str] = Field(
        description='Please enter the names of celebrity you want to train '
                    'model on. CURRENTLY NOT WORKING. WILL NOT WORK FOR DEMO.',
        example='["Demet Ozdemir", "Cristiano Ronaldo"]',
        min_items=1
    )


class DeleteCelebrityEmbeddingsCelebNamesModel(BaseModel):
    celebNames: List[str] = Field(
        description='Please enter the names of celebrity you want to train '
                    'model on.',
        example='["Demet Ozdemir", "Cristiano Ronaldo"]',
        min_items=1
    )
