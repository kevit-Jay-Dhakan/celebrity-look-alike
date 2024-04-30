import uuid
from os import makedirs
from os.path import join
from shutil import make_archive, rmtree

from pandas import read_parquet

from libs.utils.common.src.modules.string_helpers import apply_unidecode


def generate_images_of_celebs_from_dataset(celeb_list: list):
    dataset_path = 'libs/utils/ml_model/celebs_dataset.parquet'
    output_dir = f'temp/celeb_images-{uuid.uuid4()}'
    rmtree(output_dir, ignore_errors=True)
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
        make_archive(output_dir, 'zip', output_dir)
        rmtree(output_dir, ignore_errors=True)
        return print(
            f'Total {len(images_path)} images downloaded successfully.'
        )
    return 'None of the celebrity from your list is present in the dataset.'


if __name__ == '__main__':
    celeb_names = [
        "Hande Ercel",
        "Afra Saracoglu",
        "Kerem Bursin",
        "Cansu Dere",
        "Ibrahim Celikkol",
        "Neslihan Atagul",
        "Burak Ozcivit",
        "Neslihan Yeldan",
        "Akin Akinozu",
        "Esra Bilgic",
        "Bige Onal",
        "Engin Akyurek",
        "Burcu Ozberk",
        "Alp Navruz",
        "Nur Fettahoglu",
        "Bensu Soral",
        "Hazal Filiz Kucukkose",
        "Rabia Soyturk",
        "Aras Bulut Iynemli",
        "Ece Cesmioglu",
        "Busra Pekin",
        "Melis Sezen",
        "Kaan Urgancioglu",
        "Birce Akalay",
        "Sukru Ozyildiz",
        "Cemre Baysel",
        "Can Yaman",
        "Hayal Koseoglu",
        "Onur Tuna",
        "Senay Gurler",
        "Kivanc Tatlitug",
        "Belcim Bilgin",
        "Alperen Duymaz",
        "Ayca Aysin Turan",
        "Cagatay Ulusoy",
        "Hilal Altinbilek",
        "Hande Soral",
        "Hulya Darcan",
        "Dilara Aksuyek",
        "Hazal Kaya",
        "Tuba Buyukustun",
        "Elcin Sangu",
        "Melisa Dongel",
        "Ozge Ozpirincci",
        "Caner Cindoruk",
        "Beren Saat",
        "Serkan Cayoglu",
        "Cengiz Coskun",
        "Ozge Gurel",
        "Elcin Afacan",
        "Turkan Soray",
        "Selim Bayraktar",
        "Sevval Sam",
        "Serkan Altunorak",
        "Ismail Ege Sasmaz",
        "Pinar Deniz",
        "Gokhan Alkan",
        "Duygu Yetis",
        "Demet Ozdemir",
        "Engin Altan Duzyatan",
        "Kenan Ece",
        "Damla Sonmez",
        "Mert Yazicioglu",
        "Burak Deniz",
        "Burak Celik",
        "Eda Ece",
        "Irem Helvacioglu",
        "Hazal Turesan",
        "Berk Atan",
        "Hafsanur Sancaktutan",
        "Sibel Tascioglu",
        "Baris Arduc",
        "Sitare Akbas",
        "Birkan Sokullu",
        "Caglar Ertugrul",
        "Hande Ataizi",
        "Rojda Demirer",
        "Cihan Ercan",
        "Caner Topcu",
        "Evrim Dogan",
        "Miray Daner",
        "Recep Usta",
        "Defne Kayalar",
        "Mine Tugay",
        "Melike Ipek Yalova",
        "Mujde Ar",
        "Murat Aygen",
        "Nurettin Sonmez",
        "Nehir Erdogan",
        "Hazal Subasi",
        "Asli Enver",
        "Didem Balcin",
        "Ruzgar Aksoy",
        "Hatice Aslan",
        "Kenan Imirzalioglu",
        "Ekin Mert Daymaz",
        "Seda Bakan",
        "Kaan Tasaner",
        "Sevcan Yasar",
        "Farah Zeynep Abdullah",
        "Ipek Tenolcay",
        "Vahide Percin",
        "Meryem Uzerli",
        "Halit Ozgur Sari",
        "Boran Kuzum",
        "Uraz Kaygilaroglu",
        "Ozge Torer",
        "Aybuke Pusat",
        "Serpil Tamur",
        "Oznur Serceler",
        "Sebnem Donmez",
        "Fatma Toptas",
        "Yildiz Cagri Atiksoy",
        "Yeliz Kuvanci",
        "Acelya Topaloglu",
        "Metin Akdulger",
        "Cuneyt Arkin",
        "Ozan Dolunay",
        "Halit Ergenc",
        "Bugra Gulsoy",
        "Serkay Tutuncu",
        "Kubilay Aka",
        "Haluk Bilginer",
        "Binnur Kaya",
        "Hande Dogandemir",
        "Gulcin Santircioglu",
        "Ekin Koc",
        "Gokce Bahadir",
        "Alina Boz",
        "Saygin Soysal",
        "Nilperi Sahinkaya",
        "Birand Tunca",
        "Merve Dizdar",
        "Mehmet Korhan Firat",
        "Oyku Karayel",
        "Cem Yigit Uzumoglu",
        "Dilan Cicek Deniz",
        "Nurgul Yesilcay",
        "Ferit Aktug",
        "Ipek Filiz Yazici",
        "Kadir Dogulu",
        "Sevda Erginci",
        "Tuan Tunali",
        "Nesrin Cavadzade",
        "Gokce Yanardag",
        "Furkan Andic",
        "Funda Eryigit",
        "Zuhal Olcay",
        "Ozcan Deniz",
        "Salih Bademci",
        "Bulent Polat",
        "Nur Surer",
        "Nursel Kose",
        "Murat Boz",
        "Merve Cagiran",
        "Ugur Gunes",
        "Ibrahim Selim",
        "Tayanc Ayaydin",
        "Ege Kokenli",
        "Canan Erguder",
        "Bora Cengiz",
        "Tolgahan Sayisman",
        "Gunay Karacaoglu",
        "Nebahat Cehre"
    ]
    generate_images_of_celebs_from_dataset(celeb_names[:2])
