import fnmatch
import os
import re
import shutil
import smtplib
import subprocess
import time
from email.header import Header
from email.mime.text import MIMEText
from typing import Any, Dict, List, Set, Tuple
from os.path import join, exists, expanduser
import warnings
import yaml
from argparse import ArgumentParser
import gpustat

def send_mail(email: str, subject: str, content: str):
    # 第三方 SMTP 服务
    mail_host = "smtp.163.com"  #设置服务器
    mail_user = "wu_admin_audiolab@163.com"  #用户名
    mail_pass = "YCODOBGJQPFWNXPB"  #口令
    sender = 'wu_admin_audiolab@163.com'

    receivers = [email]  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(sender, 'utf-8')
    message['To'] = Header("测试", 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)  # 25 为 SMTP 端口号
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e)
        print("Error: 无法发送邮件")


def get_gpu_usage_info() -> Tuple[str, Dict[str, Dict[str, Any]]]:
    # 获取当前服务器GPU信息
    obj = subprocess.Popen(["gpustat -cpu"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    cmd_out = obj.stdout.read()  # type:ignore
    lines = cmd_out.splitlines()

    gpu_info = {}
    host_name = lines[0].split(' ')[0]
    for line in lines[1:]:
        # [1] NVIDIA A100-SXM4-80GB | 35'C,  47 % | 18013 / 81920 MB | wangyabo:python/2568107(4601M) wangyabo:python/2582338(4541M) wangyabo:python/2605856(4571M) wangyabo:python/1059186(3489M)
        match = re.match(r"\[(\d)\] .* \| .* (\d+ %) \|\s+(\d+)\s+/\s+(\d+) MB \|(.*)", line)
        gpu_id = match.group(1)  # type:ignore
        gpu_leverage = match.group(2)  # type:ignore
        gpu_men_used, gpu_mem_all = match.group(3), match.group(4)  # type:ignore
        users = match.group(5)  # type:ignore
        users = users.strip().split(" ")
        users = [user.strip().split(":")[0] for user in users if len(user) > 0]

        gpu_info[gpu_id] = {
            "利用率": gpu_leverage,
            "已用内存": int(gpu_men_used),
            "总内存": int(gpu_mem_all),
            "剩余内存": int(gpu_mem_all) - int(gpu_men_used),
            "用户": users,
        }
    return host_name, gpu_info


def get_free_gpus(gpu_info: Dict[str, Dict[str, Any]], valid_gpus: Set[str]) -> List[str]:
    free_gpus = set()
    for gpu in valid_gpus:
        info = gpu_info[gpu]
        if len(info['用户']) == 0:
            free_gpus.add(gpu)
    return list(free_gpus)


last_msg = ""


def log(msg: str):
    global last_msg
    # 去除重复的log
    if msg != last_msg:
        print(msg)
        last_msg = msg


def read_tasks(task_dir: str = "tasks") -> List[Dict[str, Any]]:
    """当前函数的作用是读取task_dir下面的配置文件。如有特殊需求，可以通过修改当前函数实现（返回值的格式应当和当前函数保持一致）。样例任务配置：

    exp_name: 训练5层神经网络
    cmd: python fit --config abc.yaml --model.exp_name={exp_name} --trainer.gpus={gpus}
    require:
        num_gpus: 1
        time_needed_to_start_completely: 60
        nohup: true
    """
    os.makedirs(task_dir, exist_ok=True)
    # 从tasks文件夹读取任务配置文件如abc.yaml
    task_files = os.listdir(task_dir)
    task_files = fnmatch.filter(task_files, "*.yaml")
    task_files.sort()  # 按照名称排序
    tasks = []
    for task_file in task_files:
        if task_file == "template.yaml":
            continue
        with open(join(task_dir, task_file), 'r') as f:
            if exists(join(task_dir, "started", task_file)):
                warnings.warn(f"{task_file}在{join(task_dir,'started')}也存在，防止出错，将忽略该任务。如需执行该任务，请删除started文件夹下面的同名文件和log。")
                continue
            task_config = yaml.safe_load(f)
            task_config['配置文件内容'] = yaml.dump(task_config, allow_unicode=True)
            if 'cmd' not in task_config:
                warnings.warn(f"{task_file}中没有cmd项")
                task_config["cmd"] = ""
            # 插入默认的配置参数
            if 'require' not in task_config:
                task_config['require'] = dict()
            if 'num_gpus' not in task_config['require']:
                task_config['require']['num_gpus'] = 1
            if 'time_needed_to_start_completely' not in task_config['require']:
                task_config['require']['time_needed_to_start_completely'] = 60
            if 'exp_name' not in task_config:
                task_config['exp_name'] = "未命名"
            if 'nohup' not in task_config['require']:
                task_config['require']['nohup'] = True
            # 使用nohup指令包装用户原本的命令
            if task_config['require']['nohup'] == True and not task_config['cmd'].strip().startswith('nohup'):
                task_config["cmd"] = f"nohup {task_config['cmd']} >> {task_dir}/started/{task_file}.log 2>&1 &"
            task_config['task_file'] = task_file
            tasks.append(task_config)

    return tasks


def start_task(
    task_config: Dict[str, Any],
    host_name: str,
    free_gpus: List[str],
    task_dir: str = "tasks",
    seed_mail_to: str = "",
) -> bool:
    task_file = task_config['task_file']
    n_gpu_required = task_config['require']['num_gpus']
    exp_name = task_config['exp_name']
    cmd = task_config['cmd']

    if len(free_gpus) >= n_gpu_required:
        gpus = ",".join(free_gpus[:n_gpu_required]) + ','  # 生成GPU号，如：0,1,2,
        cmd = cmd.format(gpus=gpus, task_file_name=task_file.replace('.yaml', ''), exp_name=exp_name)
        obj = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)

        results = obj.stdout.read()  # type:ignore
        log(f"\n任务启动，服务器：{host_name}\n任务文件：{task_file}\n实验名称: {exp_name}\n指令：{cmd}\n执行指令返回值：{results}\n配置文件内容：\n{task_config['配置文件内容']}\n")
        os.makedirs(join(task_dir, 'started'), exist_ok=True)
        shutil.move(join(task_dir, task_file), join(task_dir, 'started', task_file))
        if len(seed_mail_to) > 0:
            send_mail(seed_mail_to, "任务启动", f"服务器：{host_name}\n任务文件：{task_file}\n实验名称: {exp_name}\n指令：{cmd}\n执行指令返回值：{results}\n配置文件内容：\n{task_config['配置文件内容']}\n")
        time.sleep(task_config['require']['time_needed_to_start_completely'])
        return True
    else:
        log(f"GPU数量达不到要求：{task_file} - {exp_name}， 空闲的GPU数量={len(free_gpus)}  要求的GPU数量={n_gpu_required}")
        return False


def get_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("baidu.com", 80))
    return s.getsockname()[0]


def get_who_am_i() -> str:
    obj = subprocess.Popen(['whoami'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    return obj.stdout.read().replace('\n', '')  # type:ignore


if __name__ == "__main__":
    # 示例： run_tasks.py --gpus 0 1 2 3 --email=quancs@qq.com --email_task_started --email_task_list_empty --email_task_all_ended --endless
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", help="属于自己的GPU的编号，如'0 1 2 3 4'，输入-1则表示使用全部")
    parser.add_argument("--email", type=str, help="通知邮箱", default="")
    parser.add_argument("--email_task_started", action="store_true", help="任务启动时，发送邮件")
    parser.add_argument("--email_task_list_empty", action="store_true", help="当任务列表为空时，发送邮件提醒")
    parser.add_argument("--email_task_all_ended", action="store_true", help="当给定GPU为全部空闲，且不存在待执行任务的时候，发送邮件提醒")
    parser.add_argument("--task_dir", type=str, help="任务文件夹", default="tasks")
    parser.add_argument("--endless", action="store_true", help="不断监测任务文件夹是否有新任务。不给这个参数就只读取一次当前任务文件夹里面的任务")
    args = parser.parse_args()
    print("gpus:", args.gpus)
    print("邮箱:", args.email)
    print("任务启动时，发送邮件:", args.email_task_started)
    print("当任务列表为空时，发送邮件提醒:", args.email_task_list_empty)
    print("当任务列表为空，给定GPU全部空闲时，发送邮件提醒:", args.email_task_all_ended)
    print("无限循环：", args.endless)
    print("任务文件夹:", args.task_dir)

    if len(args.gpus) == 1 and args.gpus[0] == -1:
        gpus = set([str(g) for g in range(8)])
    else:
        gpus = set([str(g) for g in args.gpus])
    task_dir = expanduser(args.task_dir)
    ip = get_ip()
    whoami = get_who_am_i()

    seed_email_for_empty_already = False
    seed_email_for_task_all_ended_already = False

    while True:
        # 读取全部GPU的信息
        host_name, gpu_info = get_gpu_usage_info()
        # 查看配置的GPU中哪些GPU是空闲的
        free_gpus = get_free_gpus(gpu_info=gpu_info, valid_gpus=gpus)
        # 读取全部的任务
        tasks = read_tasks(task_dir=task_dir)
        at_least_one_task_fail = False

        if len(tasks) > 0:
            # 当存在任务的时候，就重置标志，使得下次任务完的时候可以继续发送邮件
            seed_email_for_empty_already = False
            seed_email_for_task_all_ended_already = False

            for task in tasks:
                # 读取全部GPU的信息
                host_name, gpu_info = get_gpu_usage_info()
                # 查看配置的GPU中哪些GPU是空闲的
                free_gpus = get_free_gpus(gpu_info=gpu_info, valid_gpus=gpus)
                # 启动任务
                suc = start_task(
                    task_config=task,
                    host_name=ip,
                    free_gpus=free_gpus,
                    task_dir=task_dir,
                    seed_mail_to=args.email if args.email_task_started else "",
                )
                if suc == False:
                    at_least_one_task_fail = True
                time.sleep(10)  # 隔10秒再次检查

        # 重新获取那些信息
        host_name, gpu_info = get_gpu_usage_info()
        free_gpus = get_free_gpus(gpu_info=gpu_info, valid_gpus=gpus)
        tasks = read_tasks(task_dir=task_dir)

        if len(tasks) == 0:
            # 没有任务的时候发送邮件报告任务列表空
            if args.email_task_list_empty == True and seed_email_for_empty_already == False:
                seed_email_for_empty_already = True
                if len(args.email) > 0:
                    send_mail(args.email, "任务列表为空", f"任务列表文件夹{args.task_dir}已空，在服务器{ip}上配置的任务已经全部启动")
                    log(f"任务列表文件夹{args.task_dir}已空，在服务器{ip}上配置的任务已经全部启动")
                else:
                    log(f"任务列表文件夹{args.task_dir}已空，在服务器{ip}上配置的任务已经全部启动，但是由于没有配置邮箱，因此不发送邮件提示")
            # 没有任务且给定的GPU全部空闲【空闲指的是没有当前的用户的任务在跑】的时候就报告任务全部完成
            all_gpu_free = True
            for gpu in gpus:
                if whoami in gpu_info[gpu]["用户"]:
                    all_gpu_free = False
            if all_gpu_free and args.email_task_all_ended == True and seed_email_for_task_all_ended_already == False:
                seed_email_for_task_all_ended_already = True
                if len(args.email) > 0:
                    send_mail(args.email, "任务全部结束", f"任务列表文件夹{args.task_dir}已空，给定GPU {str(gpus)} 全部空闲，在服务器{ip}上配置的任务已经全部结束")
                    log(f"任务列表文件夹{args.task_dir}已空，给定GPU {str(gpus)} 全部空闲，在服务器{ip}上配置的任务已经全部结束")
                else:
                    log(f"任务列表文件夹{args.task_dir}已空，给定GPU {str(gpus)} 全部空闲，在服务器{ip}上配置的任务已经全部结束")

        # 非无限循环，则退出
        if not args.endless and at_least_one_task_fail == False:
            print("\n结束")
            break
        time.sleep(10)  # 隔10秒再次检查
