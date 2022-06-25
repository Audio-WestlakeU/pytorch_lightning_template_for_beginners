# 简介
一套适合初学者的pytorch lightning代码框架。

# 安装
```
pip install -r requirements.txt
```

# 生成配置文件
```
python boring.py fit --print_config=skip_null > boring.yaml
```

# 训练
```
python boring.py fit --config boring.yaml
```

## GPU训练
```
python boring.py fit --config boring.yaml --data.batch_size=[4,8] --trainer.accumulate_grad_batches=2 --trainer.gpus=0,1
```


## 恢复训练
```
python boring.py fit --config logs/BoringModel/version_x/config.yaml --ckpt_path logs/BoringModel/version_x/checkpoints/last.ckpt
```

# 测试
```
python boring.py test --config logs/BoringModel/version_x/config.yaml --ckpt_path logs/BoringModel/version_x/checkpoints/last.ckpt
```
