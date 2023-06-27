import os
import tarfile

filepath = r"G:\AudioFiles\audio_19700101_20220110_20220523_154132_0.tar.gz"
outdir = r"G:\AudioFiles\audio"


def untar(filepath, outdir):
    '''
    Untar the tar.gz file, and only keep the .flac files.
    remove the tar.gz file
    '''
    tar = tarfile.open(filepath)
    for member in tar.getmembers():
        if member.name.endswith(".flac"):
            print(member.name)
            member.name = os.path.basename(member.name)
            tar.extract(member, outdir)

    tar.close()
    #  os.remove(filepath)


if __name__ == "__main__":
    untar(filepath, outdir)
