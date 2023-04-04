#!/usr/bin/env python3
import re, subprocess, sys, os

STAGE_FILES = []
def get_all_files(real_dir, parent_dir):
    files = os.listdir(real_dir)
    for i in files:
        file_path = os.path.join(real_dir, i)
        if i in [".git", ".codebase"]:
            continue
        elif os.path.isfile(file_path):
            STAGE_FILES.append(os.path.join(parent_dir, i))
        elif os.path.isdir(file_path):
            get_all_files(file_path, os.path.join(parent_dir, i))
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
get_all_files(CUR_DIR + "/../../", "")

keywords_list = [r"\w{0,15}[\"']{1}(AKLT|AKAP|AKTP)\w{43,44}", r"cli_[a-z0-9]{16}",r".{0,15}\.?byted.org.{0,20}",r".{0,15}\.?bytedance.net.{0,20}",r".{0,20}.bytedance\.feishu\.cn.{0,50}",r"([^*<\s|:>]{0,4})(testak|testsk|ak|sk|key|token|pass|password|secret_key|access_key|auth|secretkey|accesskey|credential|secret|access)(\s{0,10}[(=:]\s{0,6}[\"']{0,1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{16,32}[\"']{0,1})"]
ignore_list_keywords = ["[^*<>]{0,6}token[^]()!<>;/@&,]{0,10}[=:].{0,1}null,", ".{0,5}pass.{0,10}[=:].{0,1}null", "passport[=:].", "[^*<>]{0,6}key[^]()!<>;/]{0,10}[=:].{0,1}string.{0,10}", ".{0,5}pass.{0,10}[=:].{0,1}string"]

def should_ignore_keywords(ignore_list_keywords, line):
    for ignore_key in ignore_list_keywords:
        pattern = re.compile(ignore_key, flags=re.I)
        if len(pattern.findall(line)) >0:
            return True
    return False


def check_commit_file(keywords_list, ignore_list_keywords, stagefiles_list, flag, cnt):
        for commit_file in stagefiles_list:
            if commit_file.endswith(('.com','.jar','.so','.msi','.esh','.appx','.appxbundle','.msu','xpi','.sys','.deb','.apk','.gif','.jpg','.png','.jpeg','.doc','.docx','.bmp','.a','.o','.mp4','.avi','.mp3','.wma','.aac','.mkv','.zip','.rar','.tar.gz','.7z','.ttf','.otf','.exe','.bin','.pkg','.iso','.svg')):
                continue
            with open(commit_file, 'r', encoding='ISO-8859-1') as f:
                f.seek(0)
                for line in f.readlines():
                    cnt += 1
                    if (should_ignore_keywords(ignore_list_keywords, line.strip()) == False):
                        for pattern in keywords_list:
                            result = re.search(pattern, line.strip(), re.I)
                            if result != None:
                                print("\033[1;36mFile \""+ commit_file + ", line, " + str(cnt) + ",\"\033[0m" + " contains sensitive information: " + "\033[1;32m" + str(result.group(0)) + "\033[0m")
                                flag += 1
            cnt = 0
        return flag


def determine_sensitive_info(flag):
    if flag > 0:
        print("=====================================================*****================================================")
        print("\033[1;31m" + "detected that commit files have some sensitive informations, please check them." + "\033[0m")
        print("=====================================================*****================================================")
        sys.exit(1)
    elif flag == 0:
        sys.exit(0)


def main():
    flag = 0
    cnt = 0
    stagefiles_list = STAGE_FILES
    if len(stagefiles_list) >= 2:
        try:
            flag = check_commit_file(keywords_list, ignore_list_keywords, stagefiles_list, flag, cnt)
        except Exception as e:
            sys.exit(0)
        determine_sensitive_info(flag)
    else:
        sys.exit(0)

if __name__=="__main__":
    main()