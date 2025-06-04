import os

# 모든 이미지의 경로를 반환 
def get_image_paths_from_dir(fdir): # fdir: str, 예시: 'datasets_path/PSHC/BBDM_input_split/train/NECT'
    flist = os.listdir(fdir)
    flist.sort()

    # 이미지 경로를 저장할 리스트 
    image_paths = []
    # 함수를 재귀적으로 호출, 파일인 경우 바로 리스트에 경로 추가
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths
