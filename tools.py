# ps -ef|grep python | tr -s ' '|cut -d' ' -f2 | xargs kill -9

# tensorboard --logdir tensorboard