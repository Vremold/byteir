#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import json

import requests

FEISHU_TOKENS = {
    "byteir_ci_group": "568196b1-1ada-429e-9582-d8990f3cbd35",
}


def send_feishu_msg(
    title, msg, msgurl="", msg_type="post", cfg_key="byteir_ci_group", access_token=""
):
    """
    :param title:
    :param msg:
    :param msgurl:
    :param msg_type: text | post
    :param cfg_key:
    :param access_token:
    :return:
    """
    tk = access_token if access_token else FEISHU_TOKENS.get(cfg_key)
    webhook = f"https://open.feishu.cn/open-apis/bot/v2/hook/{tk}"
    post_content = {
        "zh_cn": {
            "title": f"{title}",
            "content": [
                [
                    {"tag": "text", "text": f"{msg} "},
                    {"tag": "a", "text": f"{msgurl}", "href": f"{msgurl}"},
                ]
            ],
        }
    }
    payload = {
        "msg_type": f"{msg_type}",
        "content": {f"{msg_type}": post_content},
    }

    headers = {"Content-Type": "application/json"}

    response = requests.request(
        "POST", url=webhook, headers=headers, data=json.dumps(payload)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", type=str, default="ByteIR Daily CI")
    parser.add_argument("--msg", type=str, default="byteir daily ci has completed")
    args = parser.parse_args()
    send_feishu_msg(args.ci, args.msg)
