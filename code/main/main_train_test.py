import numpy as np
import torch
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_decision_transformer import run_dt
import random
import pandas as pd
import glob
import datetime
from run.run_evaluate import run_test
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_num', type=int, default=2)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--dir', type=str, default="../data1/trajectory/trajectory_data.csv")
    parser.add_argument('--test_csv', type=str, default="../data1/traffic/period-7.csv")
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--time_dim', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--expectile', type=float, default=0.99)
    parser.add_argument('--loss_report', type=int, default=100)
    parser.add_argument('--n_ctx', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_inner', type=int, default=1024)
    parser.add_argument('--activation_function', type=str, default="relu")
    parser.add_argument('--n_position', type=int, default=1024)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--budget_rate', type=float, default=1.0)
    args = parser.parse_args()

    model_param = {
        "step_num": args.step_num,
        "save_step": args.save_step,
        "dir": args.dir,
        "test_csv": args.test_csv,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "time_dim": args.time_dim,
        "batch_size": args.batch_size,
        "device": args.device,
        "expectile": args.expectile,
        "loss_report": args.loss_report,
        "budget_rate": args.budget_rate,
        "block_config": {
            "n_ctx": args.n_ctx,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_inner": args.n_inner,
            "activation_function": args.activation_function,
            "n_position": args.n_position,
            "resid_pdrop": args.resid_pdrop,
            "attn_pdrop": args.attn_pdrop,
        }
    }

    print("Settings: \n", model_param)

    time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = "./log/" + model_param["test_csv"].split("/")[1] + "_" + time_now
    with open(filename + "_param.txt", 'w') as file:
        file.write(json.dumps(model_param))
    model_param['save_dir'] = "./saved_model/DTtest_"+time_now

    print("##################Training##################")
    run_dt(device=model_param["device"], step_num=model_param["step_num"], dir=model_param["dir"],
           save_step=model_param["save_step"], model_param=model_param, batch_size=model_param["batch_size"],
           save_dir=model_param['save_dir'], loss_report=model_param['loss_report'])

    print("##################Testing##################")
    model_param["device"]="cpu"
    csv_file = model_param["test_csv"]
    pt_files = glob.glob(os.path.join(model_param['save_dir'], '*.pt'))
    pt_names = sorted([int(i.split("\\")[-1].split("/")[-1][:-3]) for i in pt_files])
    eval_result = []
    for pt_name in pt_names:
        score, score1, conversion, exc = run_test(file_path=csv_file, model_name=str(pt_name)+".pt", model_param=model_param)
        eval_result.append([pt_name, score, score1, conversion, exc])
        print("Average score of {}: {}".format(str(pt_name)+".pt", score))
        print("Average exceed rate of {}: {}".format(str(pt_name) + ".pt", exc))
        eval_result_csv = pd.DataFrame(eval_result, columns=["file", "score", "score1", "conversion", "exceed"]).sort_values(by="file")
        eval_result_csv.to_csv(filename + "_result.csv", index=False)
