#!/usr/bin/env python
# -*- coding: utf-8 -*-
PROMPT_PREFIX = "Source: "
TARGET_PREFIX = "\nTarget: "
EOS = ""  # GPT-2 eos token

def build_example(s1: str, s2: str) -> str:
    """sentence1을 조건, sentence2를 타겟으로 하는 단일 텍스트 시퀀스 구성"""
    return f"{PROMPT_PREFIX}{s1}{TARGET_PREFIX}{s2}{EOS}"

def build_prompt(s1: str) -> str:
    """추론 시 조건 프롬프트"""
    return f"{PROMPT_PREFIX}{s1}{TARGET_PREFIX}"
