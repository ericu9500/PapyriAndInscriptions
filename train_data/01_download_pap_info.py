import os
import shutil
import tempfile
from git import Repo

def download_and_prepare_folders():
    repo_url = 'https://github.com/papyri/idp.data.git'    
    base_directory_ddb = 'DDB_EpiDoc_XML'
    base_directory_hgv = 'HGV_meta_EpiDoc'
    
    ddb_dest = os.path.join(os.getcwd(), base_directory_ddb)
    hgv_dest = os.path.join(os.getcwd(), base_directory_hgv)
    
    if os.path.exists(ddb_dest) and os.path.exists(hgv_dest):
        print(f"Directories {base_directory_ddb} and {base_directory_hgv} already exist. Skipping download.")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repository from {repo_url} into {temp_dir}...")
        Repo.clone_from(repo_url, temp_dir)
        
        ddb_src = os.path.join(temp_dir, base_directory_ddb)
        hgv_src = os.path.join(temp_dir, base_directory_hgv)
        
        if not os.path.exists(ddb_dest):
            shutil.move(ddb_src, ddb_dest)
            print(f"Downloaded and moved {base_directory_ddb} to the current directory.")
        else:
            print(f"{base_directory_ddb} already exists, skipping.")

        if not os.path.exists(hgv_dest):
            shutil.move(hgv_src, hgv_dest)
            print(f"Moved {base_directory_hgv} to the current directory.")
        else:
            print(f"{base_directory_hgv} already exists, skipping.")

if __name__ == "__main__":
    download_and_prepare_folders()

