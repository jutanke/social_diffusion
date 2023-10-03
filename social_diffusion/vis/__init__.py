import shutil
from os import listdir, makedirs, remove
from os.path import isdir as isdir_local, isfile as isfile_local, join
import cv2


def to_mp4(path, delete_pngs=True, fps=25.0, nofolder=True):
    """
    creates a video file from the given images at the path
    """
    if nofolder:
        if not delete_pngs:
            raise ValueError("the folder with all pngs will be deleted!")
    if not isdir_local(path):
        raise ValueError(f"Path not exist: {path}")
    # fourcc = cv2.VideoWriter_fourcc(*"H264")
    # fourcc = cv2.VideoWriter_fourcc(*"avc1")  # avc1 == h264
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    for fname in sorted(
        [join(path, f) for f in listdir(path) if f.endswith(".png")]
    ):  # noqa E501
        im = cv2.imread(fname)
        h = im.shape[0]
        w = im.shape[1]
        if out is None:
            out = cv2.VideoWriter(join(path, "out.mp4"), fourcc, fps, (h, w))
        out.write(im)
        if delete_pngs:
            remove(fname)
    out.release()

    if nofolder:
        root_path = "/".join(path.split("/")[:-1])
        fname_target = join(root_path, path.split("/")[-1] + ".mp4")
        fname_source = join(path, "out.mp4")
        shutil.move(fname_source, fname_target)
        # from time import sleep

        # time.sleep(1)
        shutil.rmtree(path)


def create_vis_path_local(path: str):
    """
    Creates the visualization path. If the path exists
    it is being deleted gracefully and then re-created
    """
    if isdir_local(path):
        # double check that in here we ONLY have
        # visualized data and NOTHING else!
        allowed_extensions = [".png", ".mp4", ".gif"]
        for fname in listdir(path):
            fname = join(path, fname)
            if isfile_local(fname):
                fname = fname.lower()
                is_ok = False
                for ext in allowed_extensions:
                    if fname.endswith(ext):
                        is_ok = True
                        continue
                if not is_ok:
                    raise ValueError(
                        f"<save_create_vis_path> path {path} contains forbidden files: {fname}!"  # noqa E501
                    )
        # folder is deemed save to delete
        shutil.rmtree(path)
    makedirs(path)
