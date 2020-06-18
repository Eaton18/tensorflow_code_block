import os


class FileUtil:

    @staticmethod
    def get_proj_root_path():
        proj_root_path = os.path.dirname(os.path.dirname(os.getcwd()))
        return proj_root_path

    @staticmethod
    def get_datasets_root_path():
        proj_root_path = FileUtil.get_proj_root_path()
        datasets_root_path = os.path.join(proj_root_path, "datasets")
        return datasets_root_path

if __name__ == '__main__':
    proj_root_path = FileUtil.get_proj_root_path()
    datasets_root_path = FileUtil.get_datasets_root_path()

    print(datasets_root_path)
