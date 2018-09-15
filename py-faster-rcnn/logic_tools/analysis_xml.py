
import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)
add_path(this_dir)


from pascal_voc_io import PascalVocReader


def get_file_list(file_location, ext, is_fneed_ext=False):
    file_names = []
    for roots, dirs, files in os.walk(file_location):
        # only the root
        if roots != file_location:
            break

        for f_name in files:
            if f_name.endswith(ext):
                if is_fneed_ext:
                    f_name_s = f_name
                else:
                    f_name_s = os.path.splitext(f_name)[0]

                file_names.append(f_name_s)
    return file_names


def analysis_xmls(dir_xml, ext, file_names):
    dict_obj = {}
    for file_name in file_names:
        file_path = dir_xml + '/' + file_name + '.xml'
        print(file_path)
        pvr_obj = PascalVocReader(file_path)
        shapes = pvr_obj.getShapes()
        im_info = pvr_obj.get_imageInfo()
        im_w = im_info[0][0]
        im_h = im_info[0][1]
        # im_c = im_info[0][2]
        for shape in shapes:
            key = shape[0]
            left = shape[1][0]
            top = shape[1][1]
            right = shape[1][2]
            bottom = shape[1][3]

            if 1 >= left or 1 >= top:
                print('error 0 !')
                continue
            if right >= (im_w - 2) or bottom >= (im_h - 2):
                print('error 1 !')
                continue
            if left >= right or top >= bottom:
                print('error 2 !')
                continue

            if key not in dict_obj:
                dict_obj[key] = 1
            else:
                dict_obj[key] += 1
    return dict_obj


########################################################################################################################
if __name__ == '__main__':
    dir_xml = '../data/VOCdevkit2007/VOC2007/Annotations'
    ext = '.xml'

    file_names = get_file_list(dir_xml, ext, is_fneed_ext=False)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('num', len(file_names))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    dict_obj = analysis_xmls(dir_xml, ext, file_names)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('num', len(dict_obj))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for key in dict_obj:
        # print(key, dict_obj[key])
        print(key+',')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')







