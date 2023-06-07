import os
from moviepy.editor import *
from tqdm import tqdm
import demucs.separate
import shlex


def change_str_uni(input_str) :
    sp_uni = ["\"","\'"]
    for i in sp_uni :
        #print(i)
        if i in input_str :
            input_str = input_str.replace( i , "\\%s"%(i) )
    return input_str


def main() :

    work_folder = r"F:\work\video_analyze\my_work\sound"
    """
    for file in tqdm(os.listdir( work_folder )) :
        file_name = file.replace(".mp4","")
        video_paht = os.path.join(work_folder,file)
        video = VideoFileClip(video_paht)
        audio = video.audio                       # 取出聲音
        audio.write_audiofile(os.path.join(r"F:\work\video_analyze\my_work\sound",file_name+".wav"))         # 輸出聲音為 mp3
    """
    for file_name in tqdm(os.listdir( work_folder )) :
        #print(file_name)
        if ".wav" in file_name : 
            #r_list.append( file_name )
            os.system('demucs "{}"'.format(file_name))


#NEET-[ 七つのふしぎの終わるとき]Timeless time-ver2.wav,NEET-[Trinity Seven]Seven Doors-v2.wav,NEET-[ひぐらしのなく頃に]why,or why not.wav,NEET-[メガネブ！]World's   End-ver3.wav,Own littil world.wav,[Flyer][Death Parade][NEET].wav,[NEET][Be mine!][世界征服～謀略のズヴィズダー～].wav,[NEET][Infini-T Force][To be continued...].wav,[NEET][SHINY DAYS][ゆるキャン△].wav,[NEET][SHINY DAYS][ゆるキャン△]new.wav,[NEET][SOUL EATER]PAPERMOON.wav,[NEET][SWEET HURT][Happy Sugar Life].wav,[NEET][Teekyuu]ファッとして桃源郷-ver2.wav,[NEET][Your Reality][doki doki literature club].wav,[NEET][うどんの国の金色毛鞠][S.O.S].wav,[NEET][けものフレンズ][ぼくのフレンド].wav,[NEET][けものフレンズ][ようこそジャパリパークへ].wav,[NEET][ひぐらしのなく頃に 業][I believe what you said].wav,[NEET][みだらな青ちゃんは勉強ができない][WONDERFUL WONDER].wav,[NEET][やはり俺の青春ラブコメはまちがっている。続]春擬き.wav,[NEET][わかば＊ガール]初めてガールズ!.wav,[NEET][アフリカのサラリーマン][ホワイトカラーエレジー]字幕ver.wav,[NEET][インキャインパルス][あそびあそばせ].wav,[NEET][ガヴリールドロップアウト][ガヴリールドロップキック].wav,[NEET][ケムリクサ][KEMURIKUSA].wav,[NEET][ニセコイ][Rally Go Round].wav,[NEET][世界一初恋][明日、僕は君に会いに行く。].wav,[NEET][亜人][夜は眠れるかい？].wav,[NEET][俗．さよなら絶望先生][空想ルンバ].wav,[NEET][刀剣乱舞-花丸-][花丸◎日和].wav,[NEET][坂本ですが][COOLEST].wav,[NEET][夏雪ランデブー][あなたに出会わなければ ～夏雪冬花～].wav,[NEET][学園ハンサム][GET!! 夢&DREAM].wav,[NEET][幻影ヲ駆ケル太陽]trumerei.wav,[NEET][怪物王女][Blood Queen].wav,[NEET][怪談レストラン][Lost Boy].wav,[NEET][恋と選挙とチョコレート]INITIATIVE.wav,[NEET][涼宮ハルヒの憂鬱][Super Driver].wav,[NEET][焼きたて!! ジャぱん][小さな詩].wav,[NEET][私に天使が舞い降りた!][気ままな天使たち].wav,[NEET][金色のガッシュベル!!][カサブタ].wav,[NEET][魔法少女俺][NOISY LOVE POWER☆].wav,[Soul Eater Not！][monochrome][D8202][NEET].wav,[クレヨンしんちゃん バカうまっ!B級グルメサバイバル!!][RPG][NEET]Ver-2.wav,[乖離性ミリオンアーサー][Million Ways=One Destination][NEET].wav
    #demucs.separate.main([r"F:\work\video_analyze\my_work\sound\[乖離性ミリオンアーサー][Million Ways=One Destination][NEET].wav"])
    #demucs.separate.main("[Flyer][Death Parade][NEET].wav")
if __name__ == "__main__" :
    main()