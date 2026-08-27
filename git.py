#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""git_helper —— 交互式 Git 工作流助手（沙盒验证副本）"""
import subprocess
import sys

MAIN_BRANCH = "master"


def run(cmd, check=True, capture=True):
    print(f"\n$ {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    except Exception as e:
        print(f"[!] 命令执行异常: {e}")
        return None
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        if check:
            print("[!] 命令执行失败，已中止。")
        return None
    return result


def confirm(prompt="确定要执行吗？"):
    ans = input(f"\n{prompt} [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def current_branch():
    result = subprocess.run(
        "git branch --show-current", shell=True, capture_output=True, text=True
    )
    return result.stdout.strip() or "(unknown)"


def show_status():
    print("\n===== 当前仓库状态 =====")
    run("git status", check=False)
    print("\n===== 远程配置 =====")
    run("git remote -v", check=False)
    print("=========================\n")


def sync_from_upstream():
    print("\n>>> [1] 拉取原作者代码")
    status = run("git status --porcelain", check=False, capture=True)
    if status and status.stdout.strip():
        print("[!] 当前工作区有未提交的改动")
        if confirm("是否先执行 'git stash' 暂存这些改动？"):
            run("git stash push -m \"auto-stash before sync\"")
        else:
            print("已取消同步。")
            return
    steps = [
        f"git checkout {MAIN_BRANCH}",
        "git fetch upstream",
        f"git merge upstream/{MAIN_BRANCH}",
        f"git push origin {MAIN_BRANCH}",
    ]
    print("\n即将依次执行：")
    for i, s in enumerate(steps, 1):
        print(f"  {i}) {s}")
    if not confirm("确认执行以上同步步骤？"):
        print("已取消。")
        return
    for step in steps:
        if run(step) is None:
            print("[!] 同步中断，请检查上方错误信息。")
            return
    print("\n✅ 原作者代码已同步并推送。")


def push_feature():
    print("\n>>> [2] 推送自己的功能到仓库")
    if confirm("是否先基于最新的 master 创建/更新功能分支？（推荐：是）"):
        run(f"git checkout {MAIN_BRANCH}")
        run("git fetch upstream")
        run(f"git merge upstream/{MAIN_BRANCH}")
    branch = input("\n请输入功能分支名（留空则使用当前分支）: ").strip()
    if branch:
        existing = run(f"git branch --list {branch}", check=False, capture=True)
        if existing and existing.stdout.strip():
            run(f"git checkout {branch}")
            run(f"git rebase {MAIN_BRANCH}")
        else:
            run(f"git checkout -b {branch}")
    else:
        print(f"（将使用当前分支：{current_branch()}）")
    run("git status", check=False)
    if not confirm("确认将以上改动加入提交？"):
        print("已取消。")
        return
    add_mode = input("添加方式：[A] 全部 / [F] 指定文件 ? ").strip().lower()
    if add_mode == "f":
        files = input("请输入要添加的文件（空格分隔）: ").strip()
        if not files:
            print("[!] 未指定文件，已取消。")
            return
        run(f"git add {files}")
    else:
        run("git add .")
    msg = input("请输入 commit 信息（留空使用默认）: ").strip()
    if not msg:
        msg = "chore: update via git_helper"
    if run(f"git commit -m \"{msg}\"") is None:
        print("[!] 提交失败。")
        return
    target_branch = branch if branch else current_branch()
    print(f"\n准备推送分支 '{target_branch}' 到 origin ...")
    if not confirm(f"执行: git push -u origin {target_branch} ？"):
        print("已取消推送。")
        return
    run(f"git push -u origin {target_branch}")
    print("\n✅ 功能已推送。如需回馈原作者，可到 GitHub 创建 PR。")


def main():
    while True:
        print("\n========== Git 工作流助手 ==========")
        print(f"  当前分支: {current_branch()}")
        print("  1) 拉取原作者代码")
        print("  2) 推送自己的功能到仓库")
        print("  3) 查看仓库状态")
        print("  4) 退出")
        print("===================================")
        choice = input("请选择操作 [1-4]: ").strip()
        if choice == "1":
            sync_from_upstream()
        elif choice == "2":
            push_feature()
        elif choice == "3":
            show_status()
        elif choice == "4":
            print("再见！")
            sys.exit(0)
        else:
            print("[!] 无效输入，请输入 1-4。")


if __name__ == "__main__":
    main()