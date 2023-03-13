import os
import re

def main():

    data = []
    with open(r"F:\work\video_analyze\data\audio\Beelzebub-jou no Okinimesu Mama\NNNN.lrc","r",encoding="utf8") as lnc_file :
        lines = lnc_file.readlines()
        for i in lines :
            data.append(i.replace("\n","")[:-10])
    print(data)

    all_timestep = []
    for i in data :
        if len(i) == 0 : continue
        #string = "This is a string with [12:34:56] and [00:00:01] elements."
        pattern = r"\[(\d{2}):(\d{2}):(\d{2})\]"

        matches = re.findall(pattern, i)
        """
        for i in matches :
            tmp = list(map(int,i))
            all_timestep.append(tmp[0]*60+tmp[1]+tmp[2]/100)
        """
        tmp = list(map(int,matches[1]))
        all_timestep.append(tmp[0]*60+tmp[1]+tmp[2]/100)
    with open(os.path.join(r"F:\work\video_analyze\output","%s_lnc_time.txt"%("Beelzebub")),"w") as txt_file:
        print("\t".join(list(map(str,all_timestep))),file = txt_file)


if __name__ == "__main__" :
    main()