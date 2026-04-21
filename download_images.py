import os
import requests
from duckduckgo_search import DDGS

def download_datasets():
    disease_classes = [
        "healthy crop leaf isolated",
        "tomato leaf early blight disease",
        "apple leaf scab disease",
        "corn leaf rust disease",
        "potato leaf late blight disease",
        "crop leaf calcium deficiency",
        "crop leaf magnesium deficiency",
    ]
    
    base_output_dir = "data/raw/dataset"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Began gathering images into: {base_output_dir}")
    
    with DDGS() as ddgs:
        for query in disease_classes:
            folder_name = query.replace(" ", "_")
            folder_path = os.path.join(base_output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            print(f"\\n---> Running Scraper for: {query}")
            
            try:
                # Get image URLs from DuckDuckGo
                results = list(ddgs.images(query, max_results=30))
                
                downloaded_count = 0
                for rank, img_data in enumerate(results):
                    img_url = img_data.get('image')
                    if img_url:
                        try:
                            # Download image
                            response = requests.get(img_url, timeout=5)
                            if response.status_code == 200:
                                ext = img_url.split(".")[-1].split("?")[0]
                                if ext.lower() not in ['jpg', 'jpeg', 'png']:
                                    ext = 'jpg'
                                    
                                file_path = os.path.join(folder_path, f"image_{rank}.{ext}")
                                with open(file_path, 'wb') as f:
                                    f.write(response.content)
                                downloaded_count += 1
                        except Exception as e:
                            # Skip failed downloads
                            pass
                
                print(f"Successfully scraped {downloaded_count} images into {folder_path}")
            except Exception as search_error:
                print(f"Failed to search for {query}: {search_error}")

if __name__ == "__main__":
    download_datasets()
