
import glob
import os


def renameAllFilesUnderDir(file_dir):
    '''
    Rename all the files under a directory.
    '''
    for count, filename in enumerate(sorted(os.listdir(file_dir))):
        dst = f"{str(count).zfill(8)}.png"
        print('Source: \"{}\", destination: \"{}\"'.format(filename, dst))
        src =f"{file_dir}/{filename}"
        dst =f"{file_dir}/{dst}"
        os.rename(src, dst)

        
def getFilesOfType(file_dir, type_list=['*.jpg', '*.png']):
    file_list = []
    for file_type in type_list:
        file_list.extend(glob.glob(os.path.join(file_dir, file_type)))
    return file_list
