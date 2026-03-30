import os
import os.path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

def download_file(file_url, folder):
    """
    Args:
        file_url: file url that we want to download from the internet
        folder: folder in our local computer where we want to store it
    """
    file_name_save = os.path.join(folder, os.path.basename(file_url))
    if os.path.exists(file_name_save):
        return
    try:
        # Send a GET request to download the file
        file_response = requests.get(file_url, timeout=30)

        # Save the file to the directory if the request was successful
        if file_response.status_code == 200:
            with open(file_name_save, 'wb') as file:
                file.write(file_response.content)
            print(f"Downloaded: {file_name_save}")
        else:
            print(f"Failed to download: {file_url} (Status code: {file_response.status_code})")

    except requests.RequestException as e:
        print(f"Error {file_url}: {e}")

def download_scenes():
    """
    Downloads all the scenes in a local folder.
    """
    folder = 'scenes'
    base_url = 'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes/'
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    file_links = [urljoin(base_url, link['href']) for link in soup.find_all('a', href=True)
                  if link['href'].lower().endswith('.nef')]
    os.makedirs(folder, exist_ok=True)
    # Download all scenes in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(download_file, file_links, repeat(folder))

def download_shots():
    """
    Downloads the shots for each scene.
    """
    # Base URL of the folder
    folders = ['A&W1', 'A&W2', 'A&W3', 'Harbour1', 'Harbour2', 'Harbour3',
               'Harbour4', 'SFU_art', 'blue_ceiling', 'dining_area',
               'downtown_smith', 'edu_area', 'foodcourt_mcnz', 'hallway',
               'image_theater', 'owl_statue', 'playground', 'rugs', 'seat_rows',
               'study_area', 'stump', 'subway1', 'subway2', 'theater', 'tree_tunel',
               'uncle_fatih1', 'uncle_fatih2', 'under_tree2', 'wall_art',
               'wall_hallway', 'wall_lab']

    for folder in folders:
        # Folder url link
        base_url = 'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes_shots/' + folder + '/'

        # Send a GET request to the URL
        response = requests.get(base_url)

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all anchor tags (<a>) that contain file links and filter only nef files
        # file_links = soup.find_all('a', href=True)
        file_links = [urljoin(base_url, link['href']) for link in soup.find_all('a', href=True)
         if link['href'].lower().endswith('.nef')]

        # Download files in parallel using ThreadPoolExecutor
        folder = 'scenes_shots/' + folder

        # Create a directory to save files
        os.makedirs(folder, exist_ok=True)

        max_workers = 64
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(download_file, file_links, repeat(folder))

        print(f"All files have been downloaded. Folder: {folder}")

if __name__ == '__main__':
    print('Download of scenes starting')
    download_scenes()
    print('Download of scenes finished')

    print('Download of shots starting')
    download_shots()
    print('Download of shots finished')
