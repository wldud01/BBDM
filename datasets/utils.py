import os

# 모든 이미지의 경로를 반환 
def get_image_paths_from_dir(fdir): # fdir: str, 예시: 'datasets_path/PSHC/BBDM_input_split/train/NECT'
    flist = os.listdir(fdir)
    flist.sort()
    
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths
