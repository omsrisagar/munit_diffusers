import os
from pathlib import Path
# import blobfile as bf

def _list_image_files_recursively(data_dir, return_full_paths=False):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        entry = os.path.join(data_dir, entry) if return_full_paths else entry
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(entry)
        # elif bf.isdir(full_path):
        #     results.extend(_list_image_files_recursively(full_path))
    return results

def main():
    data_dir = 'from_unsplash/images-facade/images-facade_img'
    path_r = os.path.join(data_dir, 'download_list_facade.txt')
    path_w = os.path.join(data_dir, 'metadata.jsonl')
    all_filenames = _list_image_files_recursively(data_dir)

    file_r = open(path_r, 'r')
    file_w = open(path_w, 'w')

    lines_r = file_r.readlines()
    file_r.close()
    lines_w = []
    mydict = {}

    for i in range(0, len(lines_r),2):
        url = lines_r[i]
        alt = lines_r[i+1]
        filename = os.path.basename(url[4:].strip())
        text = alt[4:].strip()
        text = text.replace('\\', '') #remove stray \ characters which are not allowed in json format
        # line_w.replace('//', '')
        mydict.update({filename: text})

    for filename in all_filenames:
        fn_key = filename.split('.')[0]
        text = mydict[fn_key]
        line_w = f'{{\"file_name\": \"{filename}\", \"text\": \"{text}\"}}\n'
        lines_w.append(line_w)

    file_w.writelines(lines_w)
    file_w.close()


if __name__ == "__main__":
    main()
