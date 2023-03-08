import yaml
import os


def main():
    base_path = r"F:\work\video_analyze\output\cut_video_data\Detective Conan The Culprit Hanzawa"
    with open( os.path.join(base_path,"paragraph_dict.yaml"),"r" ) as yamlfile :
        paragraph_dict = yaml.load(yamlfile,Loader=yaml.Loader)

    video_data_list = [i.replace(".map",".yaml") for i in os.listdir(r"F:\work\video_analyze\data\video\Detective Conan The Culprit Hanzawa") ]

    print(paragraph_dict)
    print(video_data_list)


if __name__ == "__main__" :
    main()