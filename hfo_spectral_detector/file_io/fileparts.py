import os

def fileparts(filename):
    path = os.path.dirname(os.path.abspath(__file__))
    cutIdx = path.rfind(os.path.sep)
    workspacePath = path[:cutIdx]

    images_path = workspacePath + os.path.sep + 'Images' + os.path.sep
    os.makedirs(images_path, exist_ok=True)

    return images_path