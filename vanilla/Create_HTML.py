import file2html as fh
import os
import sys
import Combine_html as ch

def create_single_html(workdir):

    f = open(workdir + "/target_name", "r")
    text = f.read()
    f.close()



    name_list = os.listdir(workdir + "/ATGO_PLUS/")
    for name in name_list:
        fh.file2html(workdir + "/ATGO_PLUS/" + name, text.splitlines()[0])
    ch.combine_html(workdir)

    for name in name_list:
        os.system("rm -rf " + workdir + "/ATGO_PLUS/" + name + "/index.html")


if __name__ == '__main__':

    create_single_html(sys.argv[1])


